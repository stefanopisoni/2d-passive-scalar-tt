"""Decaying passive scalar in a steady cellular flow: dense truth and hybrid TT.

Physics
-------
A steady, incompressible cellular flow stirs an initially large-scale scalar,

    d_t theta + u . grad theta = kappa lap theta,
    u = (d_y psi, -d_x psi),   psi = sin(x) sin(y)   on [0, 2pi)^2.

The cells wrap the initial condition into filaments, driving variance from the
injection scale down to the Batchelor scale ell_B ~ sqrt(kappa / gamma), where
gamma ~ |grad u| ~ 1 is the strain rate.  Diffusion then removes it, so without
forcing <theta^2> decays monotonically: this is a *decaying* cascade, the
transient regime, not the statistically steady state the paper's statistics
describe.  The variance budget is

    d/dt <theta^2>/2 = -kappa <|grad theta|^2>,

with no production term, and the right-hand side is what scalar_dissipation()
returns.

What this measures
------------------
The question the run answers is whether the TT rank of the solution stays
bounded once fine scales exist.  Rank is reported two ways:

  * the *intrinsic* rank of the dense (ground-truth) solution, obtained by
    recompressing the exact field at a fixed accuracy -- a property of the
    dynamics, with no solver feedback;
  * the rank the hybrid solver actually carries, which can differ because
    truncation feeds back into the dynamics.

Both are reported against the geometric bound at each cut, because a rank
measurement is vacuous if the ranks have saturated that bound: it then reflects
the grid, not the field.  Two earlier attempts at this measurement were
degenerate for exactly that reason.

Usage
-----
    python run_decaying_cellular.py --dense                  # truth only, ~1 min
    python run_decaying_cellular.py --eps 1e-6 --bd 128      # converged TT
    python run_decaying_cellular.py --bd 32                  # rank-starved

Each invocation writes one .npz into results/ ; plot_decaying_cellular.py turns
them into figures.  Hybrid runs step the dense reference alongside so that the
error history is self-contained.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hybrid_ops as ho
import hybrid_operators as hop
import hybrid_evolve as hev
from spectral_tools import compute_isotropic_spectrum_2d

RESULTS = os.path.join(_HERE, "results")

# accuracies at which the intrinsic rank of the true solution is probed
EPS_PROBE = (1e-6, 1e-4, 1e-2)


# --------------------------------------------------------------------------
# setup
# --------------------------------------------------------------------------
def grid(L: int):
    """Coordinate arrays in (row, col) = (y, x) layout."""
    n = 2 ** L
    dx = 2 * np.pi / n
    c = np.arange(n) * dx
    XX, YY = np.meshgrid(c, c, indexing="xy")   # XX varies along cols, YY along rows
    return n, dx, XX, YY


def initial_field(L: int, kmax: int = 2, seed: int = 7) -> np.ndarray:
    """Random superposition of modes with |k| <= kmax, zero mean, unit variance.

    A low-k random field rather than a single harmonic: products of sines are
    separable and therefore exactly rank-1 or rank-2 in the TT encoding, which
    would make the initial rank unrepresentatively small and the growth
    unrepresentatively dramatic.
    """
    n, dx, XX, YY = grid(L)
    rng = np.random.default_rng(seed)
    f = np.zeros((n, n))
    for kx in range(-kmax, kmax + 1):
        for ky in range(-kmax, kmax + 1):
            if not 0 < kx * kx + ky * ky <= kmax * kmax:
                continue
            f += np.cos(kx * XX + ky * YY + rng.uniform(0, 2 * np.pi))
    f -= f.mean()
    return f / f.std()


def geometric_bond_bounds(L: int, N: int):
    """Largest bond dimension permitted at each cut of the hybrid chain.

    The head fuses the first N legs, so the first cut sees 2^N on its left and
    2^(2L-N) on its right; each subsequent cut moves one leg across.
    """
    n_legs = 2 * L
    return np.array([min(2 ** cut, 2 ** (n_legs - cut))
                     for cut in range(N, n_legs)], dtype=np.int64)


def build_velocity(L: int, N: int, dx: float):
    """(ux, uy) from psi = sin x sin y, via the solver's own derivative MPOs.

    Differentiating the streamfunction discretely makes the velocity
    divergence-free to round-off (measured: 7e-13), which is what the skew form
    needs in order to conserve variance exactly.  Sampling sin/cos analytically
    instead leaves an O(dx^2) divergence.  The truncation is tight and the
    result has bond dimension 4.
    """
    _, _, XX, YY = grid(L)
    psi = ho.field_to_hybrid(torch.from_numpy(np.sin(XX) * np.sin(YY)),
                             L, N, bd_first=None, eps=1e-13)
    tight = hev.Truncation(eps=1e-12, bd_first=16, bd_remaining=16)
    return hev.velocity_from_streamfunction(psi, L, N, dx, trunc=tight)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def spectrum(f: np.ndarray):
    k, E = compute_isotropic_spectrum_2d(torch.from_numpy(np.ascontiguousarray(f)))
    return k.numpy(), E.numpy()


def field_diagnostics(f: np.ndarray, kappa: float, dx: float, L: int, N: int):
    """Scalar diagnostics of one snapshot, plus its variance spectrum."""
    k, E = spectrum(f)
    tot = E.sum()
    cum = np.cumsum(E) / tot
    gx = hev._apply_stencil_dense(f, hop.D1_CENTRAL_2, hop.AXIS_X) / dx
    gy = hev._apply_stencil_dense(f, hop.D1_CENTRAL_2, hop.AXIS_Y) / dx
    g2 = gx ** 2 + gy ** 2
    return dict(
        E=E,
        variance=float(f.var()),
        mass=float(f.mean()),
        chi=float(kappa * g2.mean()),
        grad2=float(g2.mean()),
        # centroid and the 95%-of-variance wavenumber: where the variance sits
        kbar=float((k * E).sum() / tot),
        k95=float(k[np.searchsorted(cum, 0.95)]),
        # resolution check: variance density at the grid cutoff vs at the peak
        tail_ratio=float(E[-1] / E.max()),
        # intermittency: filaments make the gradient field heavy-tailed
        grad_flatness=float((gx ** 4).mean() / (gx ** 2).mean() ** 2),
        seam=hev.seam_jump_ratio(f, L, N),
    )


def intrinsic_ranks(f: np.ndarray, L: int, N: int):
    """Bond profile of the exact field recompressed at each probe accuracy."""
    out = {}
    tf = torch.from_numpy(np.ascontiguousarray(f))
    for eps in EPS_PROBE:
        h = ho.field_to_hybrid(tf, L, N, bd_first=None, eps=eps)
        tag = f"{eps:g}"
        out[f"intrinsic_bonds_{tag}"] = np.array(ho.bond_dims(h), dtype=np.int64)
        out[f"intrinsic_params_{tag}"] = ho.num_params(h)
    return out


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def run(L: int, N: int, kappa: float, T: float, cfl: float, diff_number: float,
        bd: int, eps: float, dense_only: bool, scheme: str, form: str,
        n_samples: int, n_snaps: int, label: str):
    n, dx, _, _ = grid(L)
    dt = hev.stable_dt(1.0, 1.0, kappa, dx, cfl=cfl, diff_number=diff_number)
    n_steps = int(round(T / dt))
    every = max(1, n_steps // n_samples)
    snap_every = max(1, n_steps // max(n_snaps - 1, 1))

    f0 = initial_field(L)
    dense = f0.copy()

    # The dense reference must be driven by *exactly* the velocity the hybrid
    # solver sees, which is the streamfunction differentiated discretely -- not
    # the analytic sin/cos.  The two differ by only 1e-4 pointwise, but that
    # difference drives the two solutions apart linearly in time: measured, it
    # reaches 9.3e-4 by t=16, which is ~100x the adaptive run's truncation error
    # and would masquerade as a steadily *drifting* truncation error.
    ux, uy = build_velocity(L, N, dx)
    uxd = ho.hybrid_to_field(ux, L).numpy()
    uyd = ho.hybrid_to_field(uy, L).numpy()
    d_rhs = hev.dense_variable_rhs(uxd, uyd, dx, kappa=kappa, form=form)

    if dense_only:
        h = trunc = h_rhs = None
    else:
        trunc = hev.Truncation(eps=eps, bd_first=bd, bd_remaining=bd)
        h_rhs = hev.variable_rhs(ux, uy, L, N, dx, kappa=kappa, form=form,
                                 trunc=trunc)
        h = ho.field_to_hybrid(torch.from_numpy(f0), L, N,
                               bd_first=bd, bd_remaining=bd, eps=eps)

    rows, snaps, snap_t = [], [], []
    t0 = time.time()
    for i in range(n_steps + 1):
        if i % every == 0 or i == n_steps:
            row = dict(t=i * dt)
            row.update({f"ref_{k}": v for k, v in
                        field_diagnostics(dense, kappa, dx, L, N).items()})
            row.update(intrinsic_ranks(dense, L, N))
            if h is not None:
                fh = ho.hybrid_to_field(h, L).numpy()
                row.update({f"tt_{k}": v for k, v in
                            field_diagnostics(fh, kappa, dx, L, N).items()})
                row["tt_bonds"] = np.array(ho.bond_dims(h), dtype=np.int64)
                row["tt_params"] = ho.num_params(h)
                scale = max(np.abs(dense).max(), 1e-300)
                nrm = max(np.linalg.norm(dense), 1e-300)
                row["err_inf"] = float(np.abs(fh - dense).max() / scale)
                row["err_l2"] = float(np.linalg.norm(fh - dense) / nrm)
                # Control: compress the *true* field at this same budget. The
                # gap between this and err_* is the part of the error that comes
                # from evolving in the representation rather than from the
                # representation itself -- the paper measured the latter, this
                # experiment is about the former.
                hr = ho.field_to_hybrid(torch.from_numpy(
                    np.ascontiguousarray(dense)), L, N, bd_first=bd,
                    bd_remaining=bd, eps=eps)
                fr = ho.hybrid_to_field(hr, L).numpy()
                row["repr_err_inf"] = float(np.abs(fr - dense).max() / scale)
                row["repr_err_l2"] = float(np.linalg.norm(fr - dense) / nrm)
                row["repr_params"] = ho.num_params(hr)
            rows.append(row)
        if (i % snap_every == 0 or i == n_steps) and len(snaps) < n_snaps:
            snaps.append(dense.copy() if h is None
                         else ho.hybrid_to_field(h, L).numpy())
            snap_t.append(i * dt)
        if i == n_steps:
            break
        if h is not None:
            h = hev.STEPPERS[scheme](h, h_rhs, dt, trunc)
        dense = hev.dense_step(dense, d_rhs, dt, scheme)
    wall = time.time() - t0

    out = dict(L=L, N=N, kappa=kappa, dt=dt, dx=dx, T=T, n_steps=n_steps,
               scheme=scheme, form=form, label=label, wall=wall,
               bd=-1 if bd is None else bd, eps=-1.0 if eps is None else eps,
               dense_only=dense_only,
               bond_bound=geometric_bond_bounds(L, N),
               n_cells=4 ** L, snaps=np.array(snaps), snap_t=np.array(snap_t),
               eps_probe=np.array(EPS_PROBE))
    keys = sorted({k for r in rows for k in r})
    for k in keys:
        out[k] = np.array([r[k] for r in rows])
    if not dense_only:
        out["trunc_calls"] = trunc.n_calls

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"decay_{label}.npz")
    np.savez_compressed(path, **out)
    print(f"[{label}] {n_steps} steps, dt={dt:.5f}, {wall:.0f}s "
          f"({1000*wall/max(n_steps,1):.0f} ms/step) -> {path}")
    if not dense_only:
        print(f"[{label}] final err_inf={out['err_inf'][-1]:.2e} "
              f"max bond={out['tt_bonds'][-1].max()} "
              f"rho={out['tt_params'][-1]/4**L:.4f}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--N", type=int, default=6)
    p.add_argument("--kappa", type=float, default=5e-3)
    p.add_argument("--T", type=float, default=16.0)
    p.add_argument("--cfl", type=float, default=0.2)
    p.add_argument("--diff-number", type=float, default=0.1)
    p.add_argument("--bd", type=int, default=None)
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--dense", action="store_true", help="ground truth only")
    p.add_argument("--scheme", default="ssprk3")
    p.add_argument("--form", default="skew")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--snaps", type=int, default=5)
    p.add_argument("--label", default=None)
    a = p.parse_args()
    label = a.label or ("dense" if a.dense else
                        f"bd{a.bd}" if a.eps is None else f"eps{a.eps:g}")
    run(a.L, a.N, a.kappa, a.T, a.cfl, a.diff_number, a.bd, a.eps, a.dense,
        a.scheme, a.form, a.samples, a.snaps, label)


if __name__ == "__main__":
    main()
