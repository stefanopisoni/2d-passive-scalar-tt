"""Kraichnan passive scalar: dense runs and hybrid TT runs at the stationary state.

The model, its renewal discretisation and the lattice benchmark are described in
kraichnan.py. Each interval draws one velocity mode and one forcing mode from a
seeded generator; the dense reference and the hybrid solver consume the same
draws, so the error of the hybrid run is pathwise. For a given velocity the
equation is linear in theta and the dense dynamics never amplifies a
difference, so any growth of that error is the truncation's own.

    # dense: validation against the lattice spectrum, and spin-up
    python run_kraichnan.py --L 8 --xi 2 --dense --T 200 --save-final spin.npy
    # hybrid, started from the stationary dense field
    python run_kraichnan.py --L 8 --xi 2 --N 8 --bd 16 --init spin.npy --seed 2 --T 20

Writes results/decay_<label>.npz with the keys of run_decaying_cellular.py plus
the model parameters, time-averaged 2-D spectra, and at every sample the
increment statistics of the paper for both fields (ref_/tt_ sf, pdf, sigma_r).
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

# single-threaded: see run_decaying_cellular.py (29x slower with 4 threads)
torch.set_num_threads(int(os.environ.get("PSCALAR_THREADS", "1")))
torch.set_default_dtype(torch.float64)

import hybrid_evolve as hev  # noqa: E402
import hybrid_ops as ho  # noqa: E402
import kraichnan as kr  # noqa: E402
import run_decaying_cellular as rdc  # noqa: E402

RESULTS = os.environ.get("PSCALAR_RESULTS", os.path.join(_HERE, "results"))


def power(f: np.ndarray) -> np.ndarray:
    """|theta_hat_k|^2 with theta_hat = fft2(theta)/n^2, the normalisation of kraichnan.py."""
    return np.abs(np.fft.fft2(f) / f.size) ** 2


# Increment statistics of the paper: S_p(r) = <|delta_r theta|^p>, the flatness
# S_4/S_2^2, and the PDFs of delta_r theta / sigma_r with each field normalised
# by its own sigma_r. Increments are taken along x and y and pooled.
P_SF = (2, 4, 6)
R_PDF = (1, 2, 4, 8)
PDF_EDGES = np.linspace(-30.0, 30.0, 601)


def separations(n: int) -> np.ndarray:
    """~20 log-spaced r up to n/2, always including the powers of two."""
    geo = np.round(np.geomspace(1, n // 2, 20)).astype(int)
    return np.unique(np.concatenate([geo, 2 ** np.arange(int(np.log2(n)))]))


def increments(f: np.ndarray, r: int) -> np.ndarray:
    return np.concatenate([(np.roll(f, -r, axis=1) - f).ravel(),
                           (np.roll(f, -r, axis=0) - f).ravel()])


def increment_stats(f: np.ndarray, r_sf: np.ndarray) -> dict:
    sf = np.empty((len(P_SF), len(r_sf)))
    for j, r in enumerate(r_sf):
        d2 = increments(f, r) ** 2
        sf[:, j] = d2.mean(), (d2 * d2).mean(), (d2 ** 3).mean()
    pdf = np.empty((len(R_PDF), len(PDF_EDGES) - 1), dtype=np.int64)
    sigma = np.empty(len(R_PDF))
    for j, r in enumerate(R_PDF):
        d = increments(f, r)
        sigma[j] = d.std()
        pdf[j] = np.histogram(d / max(sigma[j], 1e-300), PDF_EDGES)[0]
    return dict(sf=sf, pdf=pdf, sigma_r=sigma)


def compress(f, L, N, bd, consistent):
    """Static compression of a dense field. With consistent, the result is
    projected onto the mass and norm of f (hybrid_evolve.project_invariants): the
    budget rule applied to a single compression, which for the norm is the
    renormalisation of the compression paper."""
    h = ho.field_to_hybrid(torch.from_numpy(np.ascontiguousarray(f)), L, N,
                           bd_first=bd, bd_remaining=bd)
    if consistent:
        h = hev.project_invariants(h, L, N, float(f.sum()), float((f ** 2).sum()))[0]
    return h


def tt_row(h, dense, L, N, bd, kappa, dx, consistent):
    """Error of the hybrid field against the dense one, and the representation
    control: the static compression of the dense field under the run's policy
    (repr_*; with the budget policy it is projected onto the mass and norm of the
    field), and the plain truncation (repr_plain_l2)."""
    fh = ho.hybrid_to_field(h, L).numpy()
    row = {f"tt_{k}": v for k, v in rdc.field_diagnostics(fh, kappa, dx, L, N).items()}
    row["tt_bonds"] = np.array(ho.bond_dims(h), dtype=np.int64)
    row["tt_params"] = ho.num_params(h)
    scale = max(np.abs(dense).max(), 1e-300)
    nrm = max(np.linalg.norm(dense), 1e-300)
    row["err_inf"] = float(np.abs(fh - dense).max() / scale)
    row["err_l2"] = float(np.linalg.norm(fh - dense) / nrm)
    hr = compress(dense, L, N, bd, consistent=False)
    fr = ho.hybrid_to_field(hr, L).numpy()
    row["repr_plain_l2"] = float(np.linalg.norm(fr - dense) / nrm)
    if consistent:
        hr = hev.project_invariants(hr, L, N, float(dense.sum()), float((dense ** 2).sum()))[0]
        fr = ho.hybrid_to_field(hr, L).numpy()
    row["repr_err_inf"] = float(np.abs(fr - dense).max() / scale)
    row["repr_err_l2"] = float(np.linalg.norm(fr - dense) / nrm)
    row["repr_params"] = ho.num_params(hr)
    return row, fh


def run(L, xi, qmax, ku, kappa, T, cfl, diff_number, N, bd, dense_only, seed,
        init, save_final, n_samples, n_snaps, t_avg, spec_every, label,
        policy="plain"):
    if policy not in ("plain", "budget"):
        raise ValueError(f"unknown policy {policy!r}")
    budgeted = policy == "budget"
    model = kr.Kraichnan(L, xi, qmax, ku)
    if kappa is None:
        kappa = model.default_kappa()
    n, dx, XX, YY = rdc.grid(L)
    tau = model.tau
    n_int = int(round(T / tau))
    every = max(1, n_int // n_samples)
    snap_every = max(1, n_int // max(n_snaps - 1, 1))
    rng = np.random.default_rng(seed)

    f0 = np.zeros((n, n)) if init is None else np.load(init)
    if f0.shape != (n, n):
        raise SystemExit(f"--init {init} has shape {f0.shape}, need {(n, n)}")
    dense = f0.copy()
    if dense_only:
        h = trunc = ops = None
    else:
        if init is None:
            raise SystemExit("hybrid runs start from a stationary dense field: pass --init")
        ops = hev.transport_operators(L, dx, kappa)
        # every truncation is plain; with the budget policy each update (step,
        # forcing increment, initial compression) is projected onto the mass and
        # variance the physics gives it (hybrid_evolve.PhysicalBudget); the state
        # then carries bond bd + 1, the extra one being the uniform mode
        trunc = hev.Truncation(bd_first=bd, bd_remaining=bd)
        budget = hev.PhysicalBudget(L, N, kappa, lap=ops[2]) if budgeted else None
        h = compress(f0, L, N, bd, budgeted)

    spec_ref = np.zeros((n, n))
    n_spec = 0
    spec_ref_s, spec_tt_s, n_spec_s = np.zeros((n, n)), np.zeros((n, n)), 0
    rows, snaps, ref_snaps, snap_t = [], [], [], []
    r_sf = separations(n)
    n_sub = 0
    t0 = time.time()
    for i in range(n_int + 1):
        t = i * tau
        if i % every == 0 or i == n_int:
            row = dict(t=t, n_sub=n_sub, wall_t=time.time() - t0)
            row.update({f"ref_{k}": v for k, v in
                        rdc.field_diagnostics(dense, kappa, dx, L, N).items()})
            row.update(rdc.intrinsic_ranks(dense, L, N))
            row.update({f"ref_{k}": v for k, v in increment_stats(dense, r_sf).items()})
            if h is not None:
                r, fh = tt_row(h, dense, L, N, bd, kappa, dx, budgeted)
                row.update(r)
                if budgeted:
                    row["budget_dev_mean"], row["budget_dev_max"] = budget.dev_mean, budget.dev_max
                    row["budget_dm_max"] = budget.dm_max
                    budget.reset()
                row.update({f"tt_{k}": v for k, v in increment_stats(fh, r_sf).items()})
                if t >= t_avg:
                    spec_ref_s += power(dense)
                    spec_tt_s += power(fh)
                    n_spec_s += 1
            rows.append(row)
        if spec_every and t >= t_avg and i % spec_every == 0:
            spec_ref += power(dense)
            n_spec += 1
        if (i % snap_every == 0 or i == n_int) and len(snaps) < n_snaps:
            snaps.append(dense.copy() if h is None else ho.hybrid_to_field(h, L).numpy())
            if h is not None:
                ref_snaps.append(dense.copy())
            snap_t.append(t)
        if i == n_int:
            break

        iv, ph, jf, phf = model.draw(rng)
        a = float(model.amp[iv])
        m = int(np.ceil(tau / hev.stable_dt(a, a, kappa, dx, cfl=cfl,
                                            diff_number=diff_number) - 1e-9))
        dt = tau / m
        uxd, uyd = model.velocity_dense(iv, ph, XX, YY)
        d_rhs = hev.dense_variable_rhs(uxd, uyd, dx, kappa=kappa, form="skew")
        for _ in range(m):
            # the dense twin follows the same policy, so the two differ by compression only
            dense = (hev.dense_step_ssprk3_budget(dense, d_rhs, dt, kappa, dx) if budgeted
                     else hev.dense_step(dense, d_rhs, dt, "ssprk3"))
        dense = dense + model.forcing_dense(jf, phf, XX, YY)
        if h is not None:
            ux, uy = model.velocity_hybrid(iv, ph, N)
            rhs = hev.variable_rhs(ux, uy, L, N, dx, kappa=kappa, form="skew",
                                   trunc=trunc, ops=ops)
            fh_inc = model.forcing_hybrid(jf, phf, N)
            if budgeted:
                for _ in range(m):
                    h = hev.step_ssprk3_budget(h, rhs, dt, trunc, budget)
                h = budget.update_with(h, fh_inc, trunc)
            else:
                for _ in range(m):
                    h = hev.step_ssprk3(h, rhs, dt, trunc)
                h = trunc(ho.hybrid_add(h, fh_inc))
        n_sub += m
    wall = time.time() - t0

    if save_final:
        os.makedirs(os.path.dirname(os.path.abspath(save_final)), exist_ok=True)
        np.save(save_final, dense)

    out = dict(L=L, N=N, kappa=kappa, dx=dx, T=T, tau=tau, n_int=n_int,
               n_sub_total=n_sub, seed=seed, label=label, wall=wall,
               bd=-1 if bd is None else bd, dense_only=dense_only, cfl=cfl,
               policy=policy,
               diff_number=diff_number, init="" if init is None else init,
               t_avg=t_avg, spec_every=spec_every,
               spec_ref=spec_ref / max(n_spec, 1), n_spec=n_spec,
               spec_ref_s=spec_ref_s / max(n_spec_s, 1),
               spec_tt_s=spec_tt_s / max(n_spec_s, 1), n_spec_s=n_spec_s,
               bond_bound=rdc.geometric_bond_bounds(L, N), n_cells=4 ** L,
               snaps=np.array(snaps), snap_t=np.array(snap_t),
               ref_snaps=np.array(ref_snaps), r_sf=r_sf, p_sf=np.array(P_SF),
               r_pdf=np.array(R_PDF), pdf_edges=PDF_EDGES,
               eps_probe=np.array(rdc.EPS_PROBE), velocity="kraichnan",
               **{f"model_{k}": v for k, v in model.describe().items()})
    for k in sorted({k for r in rows for k in r}):
        out[k] = np.array([r[k] for r in rows])
    if not dense_only:
        out["trunc_calls"] = trunc.n_calls

    os.makedirs(RESULTS, exist_ok=True)
    path = os.path.join(RESULTS, f"decay_{label}.npz")
    np.savez_compressed(path, **out)
    print(f"[{label}] xi={xi} ku={ku} tau={tau:.4g} kappa={kappa:.4g} "
          f"{n_int} intervals, {n_sub} substeps ({n_sub / max(n_int, 1):.1f}/interval), "
          f"{wall:.0f}s ({1000 * wall / max(n_sub, 1):.1f} ms/substep) -> {path}")
    print(f"[{label}] var(T)={dense.var():.4f}")
    if not dense_only:
        print(f"[{label}] final err_l2={out['err_l2'][-1]:.2e} "
              f"repr={out['repr_err_l2'][-1]:.2e} rho={out['tt_params'][-1] / 4 ** L:.4f}")
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, default=8)
    p.add_argument("--xi", type=float, required=True)
    p.add_argument("--qmax", type=float, default=None,
                   help="velocity cutoff (default: n/8 for xi<2, sqrt(2) for xi=2)")
    p.add_argument("--ku", type=float, default=0.5, help="Kubo number of an interval")
    p.add_argument("--kappa", type=float, default=None,
                   help="default: Kraichnan.default_kappa()")
    p.add_argument("--T", type=float, default=20.0)
    p.add_argument("--cfl", type=float, default=0.5)
    p.add_argument("--diff-number", type=float, default=0.1)
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--bd", type=int, default=None)
    p.add_argument("--dense", action="store_true", help="dense run only")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--init", default=None, help=".npy initial field (default: zero)")
    p.add_argument("--save-final", default=None, help="save the final dense field (.npy)")
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--snaps", type=int, default=4)
    p.add_argument("--t-avg", type=float, default=0.0,
                   help="start of the time average of the spectra")
    p.add_argument("--spec-every", type=int, default=1,
                   help="intervals between dense spectrum samples (0: off)")
    p.add_argument("--policy", choices=("plain", "budget"), default="plain",
                   help="budget: project every compressed update onto the variance "
                        "and mass budgets integrated by the stepper (hybrid_evolve.PhysicalBudget)")
    p.add_argument("--label", required=True)
    a = p.parse_args()
    qmax = a.qmax if a.qmax is not None else (np.sqrt(2.0) if a.xi >= 2 else 2 ** a.L / 8)
    run(a.L, a.xi, qmax, a.ku, a.kappa, a.T, a.cfl, a.diff_number, a.N, a.bd,
        a.dense, a.seed, a.init, a.save_final, a.samples, a.snaps, a.t_avg,
        a.spec_every, a.label, policy=a.policy)


if __name__ == "__main__":
    main()
