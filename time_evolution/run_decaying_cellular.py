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

# This workload is thousands of *small* contractions and SVDs per step, not a
# few large ones, so each operation is far too small to amortise thread
# dispatch. Measured on a 256-core cluster node: asking for 4 threads instead of
# 1 makes the hot einsum in _apply_core 43x slower (0.14 -> 5.9 ms) and the whole
# step 29x slower (613 -> 19869 ms at L=8, chi=32). One thread per process, many
# processes, is the right shape -- hence torch is pinned here rather than left to
# infer a thread count from the machine it lands on.
torch.set_num_threads(int(os.environ.get("PSCALAR_THREADS", "1")))

import hybrid_ops as ho
import hybrid_operators as hop
import hybrid_evolve as hev
from spectral_tools import compute_isotropic_spectrum_2d

# Overridable so cluster jobs can write to a scratch/run root instead of
# into the repository tree.
RESULTS = os.environ.get("PSCALAR_RESULTS", os.path.join(_HERE, "results"))

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


def forcing_field(L: int, amp: float, kf: int = 1) -> np.ndarray:
    """Steady large-scale source, f = amp * sin(kf * y).

    Without a source the scalar simply decays, so the statistics describe a
    transient. A source held fixed in time lets <theta^2> settle where
    production <theta f> balances dissipation kappa<|grad theta|^2>.
    """
    _, _, _, YY = grid(L)
    return amp * np.sin(kf * YY)


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


def build_velocity(L: int, N: int, dx: float, source: str = "cellular",
                   psi_eps: float = 1e-2, psi_path: str = None):
    """(ux, uy) from a streamfunction, via the solver's own derivative MPOs.

    ``source="cellular"`` uses psi = sin x sin y, which is exactly rank 2 and
    needs no truncation. ``source="dns"`` loads a streamfunction obtained from a
    DNS velocity snapshot (vorticity inverted spectrally, then coarsened) and
    compresses it at ``psi_eps``: a real 2-D turbulent streamfunction is close to
    full rank, saturating the geometric bound at eps = 1e-6 and costing more
    parameters than the dense field, so it is only usable in truncated form.
    The truncation is a modelling choice and is recorded in the run file.

    Differentiating the streamfunction discretely makes the velocity
    divergence-free to round-off (measured: 7e-13), which is what the skew form
    needs in order to conserve variance exactly.  Sampling sin/cos analytically
    instead leaves an O(dx^2) divergence.  The truncation is tight and the
    result has bond dimension 4.
    """
    if source == "cellular":
        _, _, XX, YY = grid(L)
        psi_d = np.sin(XX) * np.sin(YY)
        eps, cap = 1e-13, 16
    elif source == "dns":
        if psi_path is None:
            psi_path = os.path.join(RESULTS, f"psi_dns_L{L}.npy")
        psi_d = np.load(psi_path)
        if psi_d.shape != (2 ** L, 2 ** L):
            raise ValueError(f"{psi_path} is {psi_d.shape}, expected {2**L}^2")
        eps, cap = psi_eps, 256
    else:
        raise ValueError(f"unknown velocity source {source!r}")
    psi = ho.field_to_hybrid(torch.from_numpy(np.ascontiguousarray(psi_d)),
                             L, N, bd_first=None, eps=eps)
    tight = hev.Truncation(eps=1e-10, bd_first=cap, bd_remaining=cap)
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
# shared dense reference
# --------------------------------------------------------------------------
def dense_cache_paths(stem: str):
    """Fields are kept in a plain .npy so they can be memory-mapped."""
    return stem + "_fields.npy", stem + "_meta.npz"


def sample_schedule(n_steps: int, n_samples: int):
    """Step indices at which diagnostics are recorded."""
    every = max(1, n_steps // n_samples)
    idx = [i for i in range(n_steps + 1) if i % every == 0]
    if idx[-1] != n_steps:
        idx.append(n_steps)
    return every, np.array(idx, dtype=np.int64)


def load_dense_cache(stem: str, L: int, kappa: float, n_steps: int, idx):
    """Memory-map a precomputed dense trajectory, refusing a mismatched one.

    Every run at a given (L, kappa, T, samples) steps the *same* dense
    reference, and at 4096^2 that costs 56.5 h -- far more than the hybrid
    evolution it exists to validate. Computing it once and sharing it turns
    twelve 60-hour jobs into twelve 5-to-25-hour ones.
    """
    fp, mp = dense_cache_paths(stem)
    if not (os.path.exists(fp) and os.path.exists(mp)):
        return None
    meta = np.load(mp)
    ok = (int(meta["L"]) == L and int(meta["n_steps"]) == n_steps
          and np.allclose(float(meta["kappa"]), kappa, rtol=1e-12)
          and np.array_equal(meta["idx"], idx))
    if not ok:
        raise SystemExit(f"dense cache {fp} does not match this run "
                         f"(L/kappa/steps/sample grid differ) -- rebuild it")
    return np.load(fp, mmap_mode="r")


# --------------------------------------------------------------------------
# drivers
# --------------------------------------------------------------------------
def run(L: int, N: int, kappa: float, T: float, cfl: float, diff_number: float,
        bd: int, eps: float, dense_only: bool, scheme: str, form: str,
        n_samples: int, n_snaps: int, label: str,
        dense_cache: str = None, build_dense_cache: bool = False,
        forcing_amp: float = 0.0, forcing_k: int = 1,
        velocity: str = "cellular", psi_eps: float = 1e-2):
    n, dx, _, _ = grid(L)

    # The dense reference must be driven by *exactly* the velocity the hybrid
    # solver sees, which is the streamfunction differentiated discretely -- not
    # the analytic sin/cos.  The two differ by only 1e-4 pointwise, but that
    # difference drives the two solutions apart linearly in time: measured, it
    # reaches 9.3e-4 by t=16, which is ~100x the adaptive run's truncation error
    # and would masquerade as a steadily *drifting* truncation error.
    ux, uy = build_velocity(L, N, dx, source=velocity, psi_eps=psi_eps)
    uxd = ho.hybrid_to_field(ux, L).numpy()
    uyd = ho.hybrid_to_field(uy, L).numpy()

    # The timestep follows the velocity actually built. The cellular flow has
    # |u| <= 1, but a DNS snapshot does not, and taking the CFL limit from an
    # assumed unit speed would overshoot it by that factor.
    umax = float(max(np.abs(uxd).max(), np.abs(uyd).max()))
    dt = hev.stable_dt(umax, umax, kappa, dx, cfl=cfl, diff_number=diff_number)
    n_steps = int(round(T / dt))
    every, sample_idx = sample_schedule(n_steps, n_samples)
    snap_every = max(1, n_steps // max(n_snaps - 1, 1))
    cache = (load_dense_cache(dense_cache, L, kappa, n_steps, sample_idx)
             if dense_cache and not build_dense_cache else None)

    f0 = initial_field(L)
    dense = f0.copy()
    if forcing_amp:
        f_dense = forcing_field(L, forcing_amp, forcing_k)
        f_hyb = ho.field_to_hybrid(torch.from_numpy(f_dense), L, N,
                                   bd_first=None, eps=1e-13)
    else:
        f_dense = f_hyb = None
    d_rhs = hev.dense_variable_rhs(uxd, uyd, dx, kappa=kappa, form=form,
                                   forcing=f_dense)

    if build_dense_cache:
        fields = np.empty((len(sample_idx), n, n), dtype=np.float64)
        k = 0
        t0 = time.time()
        for i in range(n_steps + 1):
            if k < len(sample_idx) and i == sample_idx[k]:
                fields[k] = dense
                k += 1
            if i == n_steps:
                break
            dense = hev.dense_step(dense, d_rhs, dt, scheme)
        fp, mp = dense_cache_paths(dense_cache)
        os.makedirs(os.path.dirname(fp) or ".", exist_ok=True)
        np.save(fp, fields)
        np.savez(mp, L=L, N=N, kappa=kappa, T=T, dt=dt, n_steps=n_steps,
                 idx=sample_idx, scheme=scheme, form=form)
        print(f"[{label}] dense cache: {len(sample_idx)} samples, "
              f"{fields.nbytes/1e9:.2f} GB, {(time.time()-t0)/3600:.1f} h -> {fp}")
        return fp

    if dense_only:
        h = trunc = h_rhs = None
    else:
        trunc = hev.Truncation(eps=eps, bd_first=bd, bd_remaining=bd)
        h_rhs = hev.variable_rhs(ux, uy, L, N, dx, kappa=kappa, form=form,
                                 forcing=f_hyb, trunc=trunc)
        h = ho.field_to_hybrid(torch.from_numpy(f0), L, N,
                               bd_first=bd, bd_remaining=bd, eps=eps)

    rows, snaps, snap_t = [], [], []
    sample_k = 0
    t0 = time.time()
    for i in range(n_steps + 1):
        if i % every == 0 or i == n_steps:
            if cache is not None:
                dense = np.asarray(cache[sample_k])
            sample_k += 1
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
        if cache is None:
            dense = hev.dense_step(dense, d_rhs, dt, scheme)
    wall = time.time() - t0

    out = dict(L=L, N=N, kappa=kappa, dt=dt, dx=dx, T=T, n_steps=n_steps,
               scheme=scheme, form=form, label=label, wall=wall,
               bd=-1 if bd is None else bd, eps=-1.0 if eps is None else eps,
               dense_only=dense_only,
               forcing_amp=forcing_amp, forcing_k=forcing_k,
               velocity=velocity, psi_eps=psi_eps, umax=umax,
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
    p.add_argument("--dense-cache", default=None,
                   help="stem of a shared dense trajectory to use (or build)")
    p.add_argument("--build-dense-cache", action="store_true",
                   help="compute the dense trajectory and save it, then exit")
    p.add_argument("--forcing-amp", type=float, default=0.0,
                   help="amplitude of a steady source f = amp*sin(k y)")
    p.add_argument("--forcing-k", type=int, default=1)
    p.add_argument("--velocity", choices=("cellular", "dns"), default="cellular")
    p.add_argument("--psi-eps", type=float, default=1e-2,
                   help="truncation of the DNS streamfunction")
    a = p.parse_args()
    label = a.label or ("dense" if a.dense else
                        f"bd{a.bd}" if a.eps is None else f"eps{a.eps:g}")
    run(a.L, a.N, a.kappa, a.T, a.cfl, a.diff_number, a.bd, a.eps, a.dense,
        a.scheme, a.form, a.samples, a.snaps, label,
        dense_cache=a.dense_cache, build_dense_cache=a.build_dense_cache,
        forcing_amp=a.forcing_amp, forcing_k=a.forcing_k,
        velocity=a.velocity, psi_eps=a.psi_eps)


if __name__ == "__main__":
    main()
