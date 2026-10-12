"""Constant-velocity advection: the decisive test of hybrid TT time evolution.

Run:  python tests/test_advection.py

Structure of the argument
-------------------------
1. Translation by one cell is an exact, rank-preserving MPO, so a CFL = 1 run
   has a known exact solution and *no* discretisation error at all.  Any drift
   is therefore 100% attributable to truncation -- which is what makes this the
   test worth running first.
2. With truncation switched off entirely, the run must be exact to machine
   precision.  That separates "the algebra is wrong" from "compression hurts".
3. Only then do we turn truncation on and measure how the error, the variance,
   and the seam amplitude evolve over many steps, with the seam smoother on and
   off.
"""
import os
import sys
import warnings

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # time_evolution/
sys.path.insert(0, _PKG)
sys.path.insert(0, os.path.dirname(_PKG))  # repo root: config.py, hybrid_MPS.py

import numpy as np
import torch

import config as cfg

sys.path.append(cfg.TENSNET_SRC)

import hybrid_evolve as hev  # noqa: E402
import hybrid_operators as hop  # noqa: E402
import hybrid_ops as ho  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

FAILURES = []


def check(name, err, tol):
    good = bool(err <= tol)
    if not good:
        FAILURES.append(name)
    print(f"  {'PASS' if good else '**FAIL**':10s} {name:50s} err={err:.3e} (tol {tol:.0e})")


L, N = 6, 6            # 64 x 64, head = 8 x 8 coarse blocks, seam period 8
n = 2**L
BD = 8
rng = np.random.default_rng(2024)

# multiscale initial condition: a smooth blob plus a k^-1 random field, so there
# is genuine small-scale content for truncation to bite on
x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
blob = np.exp(-(((x - 20) ** 2 + (y - 32) ** 2) / 40.0))
kk = np.fft.fftfreq(n) * n
KX, KY = np.meshgrid(kk, kk, indexing="xy")
K = np.sqrt(KX**2 + KY**2)
K[0, 0] = 1.0
noise = np.real(np.fft.ifft2((rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * K**-1.5))
f0 = blob + 0.5 * noise / noise.std()
F0 = torch.from_numpy(f0)

print(f"grid {n}x{n}, L={L}, N={N}, coarse blocks {2**(N // 2)}x{2**(N // 2)}, "
      f"seam period {ho.seam_period(L, N)}")

# ==========================================================================
print("\n=== 1. exact translation MPO ===")
for cells in [1, 2, -1, 3]:
    for axis, nm in [(hop.AXIS_X, "x"), (hop.AXIS_Y, "y")]:
        T = hop.translation_mpo(L, axis, cells)
        h = ho.field_to_hybrid(F0, L, N, bd_first=2**N, bd_remaining=2**L, eps=1e-15)
        got = ho.hybrid_to_field(ho.hybrid_apply(T, h, N), L).numpy()
        ref = np.roll(f0, cells, axis=axis)
        check(f"translate {cells:+d} along {nm}", np.abs(got - ref).max() / np.abs(f0).max(), 1e-12)
print(f"  translation MPO bond dim: {hop.translation_mpo(L, hop.AXIS_X, 1).max_rank}")

print("\n  -- does translation preserve TT rank?  (it does not) --")
h = ho.field_to_hybrid(F0, L, N, bd_first=BD, bd_remaining=BD)
T = hop.translation_mpo(L, hop.AXIS_X, 1)
hT = ho.hybrid_apply(T, h, N)
print(f"     before          {ho.bond_dims(h)}")
print(f"     after apply     {ho.bond_dims(hT)}   (x MPO bond dim 2)")
hTc = ho.hybrid_compress(hT, bd_first=BD, bd_remaining=BD)
err_rt = np.abs(ho.hybrid_to_field(hTc, L).numpy()
                - np.roll(ho.hybrid_to_field(h, L).numpy(), 1, axis=hop.AXIS_X)).max()
print(f"     recompressing back to bd={BD} costs {err_rt / np.abs(f0).max():.3e} relative error")
print("     -> a translated rank-r field genuinely has rank > r: in the Morton/QTT")
print("        encoding a one-cell shift is carry propagation, not a local move.")
print("        Even the most benign dynamics imaginable needs lossy truncation")
print("        every step. What IS exact here is the operator and the reference")
print("        solution (np.roll), so all observed drift is still truncation alone.")
sv_growth = []
for bd_probe in [4, 8, 16, 32]:
    hp = ho.field_to_hybrid(F0, L, N, bd_first=bd_probe, bd_remaining=bd_probe)
    hpT = ho.hybrid_compress(ho.hybrid_apply(T, hp, N), bd_first=2**N, bd_remaining=2**L,
                             eps=1e-13)
    sv_growth.append((bd_probe, max(ho.bond_dims(hpT))))
print(f"     true rank after one shift: " +
      ", ".join(f"bd {a} -> {b}" for a, b in sv_growth))

# ==========================================================================
print("\n=== 2. no-truncation run must be exact ===")
h = ho.field_to_hybrid(F0, L, N, bd_first=2**N, bd_remaining=2**L, eps=1e-15)
n_steps = 64
for i in range(n_steps):
    h = ho.hybrid_apply(T, h, N)
    h = ho.hybrid_compress(h, bd_first=2**N, bd_remaining=2**L, eps=1e-14)
got = ho.hybrid_to_field(h, L).numpy()
check(f"{n_steps} exact translations == np.roll", np.abs(got - np.roll(f0, n_steps, axis=hop.AXIS_X)).max()
      / np.abs(f0).max(), 1e-11)
check("full period returns the initial field",
      np.abs(got - f0).max() / np.abs(f0).max(), 1e-11)
print(f"  (64 steps at CFL=1 on a 64-wide periodic domain is exactly one revolution)")

# ==========================================================================
print("\n=== 3. truncated advection: 256 steps = 4 revolutions ===")
h_ic = ho.field_to_hybrid(F0, L, N, bd_first=BD, bd_remaining=BD)
f_ic = ho.hybrid_to_field(h_ic, L).numpy()
ic_err = np.abs(f_ic - f0).max() / np.abs(f0).max()
print(f"  bd={BD}. Baseline: compressing the initial field already costs "
      f"{ic_err:.3e} relative error.")
print("  Runs below are measured against exact translation OF THE COMPRESSED IC,")
print("  so they show error added by time stepping, not by the initial compression.")

n_steps = 256
configs = [("seam off", None, 1, False),
           ("seam every step", "S", 1, False),
           ("seam every 32", "S", 32, False),
           ("seam off + mass", None, 1, True)]
results = {}
for tag, use_S, s_every, mass in configs:
    h = ho.field_to_hybrid(F0, L, N, bd_first=BD, bd_remaining=BD)
    S = hop.seam_smoother_mpo(L, N) if use_S else None
    trunc = hev.Truncation(bd_first=BD, bd_remaining=BD)
    mean0 = ho.hybrid_sum(h) / 4**L
    hist = hev.History()
    hev.record(hist, 0.0, h, L, N, f_ic, None)
    for i in range(1, n_steps + 1):
        h = trunc(ho.hybrid_apply(T, h, N))
        if S is not None and i % s_every == 0:
            h = trunc(ho.hybrid_apply(S, h, N))
        if mass:
            h = ho.project_mass(h, L, N, mean0)   # after truncation, never before
        if i % 64 == 0:
            hev.record(hist, float(i), h, L, N, np.roll(f_ic, i, axis=hop.AXIS_X), None)
    results[tag] = (hist, trunc)

print(f"\n  {'configuration':<18} {'step':>5} {'err':>11} {'variance':>10} {'seam':>6} {'mass drift':>11}")
for tag, _, _, _ in configs:
    hist, _ = results[tag]
    for j in range(1, len(hist.t)):
        lbl = tag if j == 1 else ""
        md = abs(hist.mass[j] - hist.mass[0]) / max(abs(hist.mass[0]), 1e-12)
        print(f"  {lbl:<18} {int(hist.t[j]):>5} {hist.err_vs_dense[j]:>11.4e} "
              f"{hist.variance[j]:>10.5f} {hist.seam[j]:>6.2f} {md:>11.2e}")

v0 = results["seam off"][0].variance[0]
print(f"\n  variance of the compressed IC: {v0:.5f}")
for tag, _, _, _ in configs:
    hist, trunc = results[tag]
    print(f"  {tag:<18} variance retained {100 * hist.variance[-1] / v0:5.1f}%   "
          f"seam {hist.seam[0]:.2f} -> {hist.seam[-1]:.2f}   "
          f"cumulative l2^2 truncated {trunc.lost_norm2:.3e}")

hoff = results["seam off"][0]
growth = hoff.err_vs_dense[-1] / max(hoff.err_vs_dense[1], 1e-30)
print(f"\n  error growth, step 64 -> 256 (4x more steps): {growth:.2f}x  "
      f"({'saturating' if growth < 2 else 'compounding'})")
if growth > 4.0:
    FAILURES.append("error growth faster than linear in step count")

mass_free = max(abs(m - hoff.mass[0]) for m in hoff.mass) / max(abs(hoff.mass[0]), 1e-12)
hfix = results["seam off + mass"][0]
mass_fixed = max(abs(m - hfix.mass[0]) for m in hfix.mass) / max(abs(hfix.mass[0]), 1e-12)
print(f"  mass drift: uncorrected {mass_free:.2e}, with project_mass {mass_fixed:.2e}")
check("project_mass restores mass conservation", mass_fixed, 1e-12)

# ==========================================================================
print("\n=== 4. advection-diffusion vs dense FD and vs the exact solution ===")
# grid units (dx = 1). Taking dt from the CFL condition alone is unstable as soon
# as the grid is fine: explicit 2-D diffusion needs dt <= dx^2/(4 kappa).
cx, cy, kappa, dx = 0.7, -0.4, 0.05, 1.0
dt = hev.stable_dt(cx, cy, kappa, dx)
n_steps = 200
op = hop.advection_diffusion_mpo(L, cx=cx, cy=cy, kappa=kappa, dx=dx)
print(f"  RHS MPO bond dim {op.max_rank}, dt={dt:.3f}, {n_steps} steps, "
      f"CFL={max(abs(cx), abs(cy)) * dt / dx:.2f}, "
      f"diffusion number={kappa * dt / dx**2:.3f}")

for bd in [8, 16, 32]:
    h0 = ho.field_to_hybrid(F0, L, N, bd_first=bd, bd_remaining=bd)
    f_start = ho.hybrid_to_field(h0, L).numpy()
    trunc = hev.Truncation(bd_first=bd, bd_remaining=bd)
    _, hist, _ = hev.evolve(
        h0, L, N, op, dt, n_steps, trunc, scheme="rk2",
        dense_f0=f_start, dense_rhs=hev.dense_rhs_adv_diff(cx, cy, kappa, dx),
        exact_fn=lambda t: hev.spectral_reference(f_start, t, cx, cy, kappa, dx),
        every=n_steps)
    print(f"  bd={bd:>2}: err vs dense-FD (TT truncation only) = {hist.err_vs_dense[-1]:.3e}   "
          f"err vs spectral (total) = {hist.err_vs_exact[-1]:.3e}   "
          f"seam {hist.seam[0]:.2f} -> {hist.seam[-1]:.2f}")

print("\n  -- dense FD scheme itself, against the exact solution (no TT involved) --")
f_d = f0.copy()
rhs_d = hev.dense_rhs_adv_diff(cx, cy, kappa, dx)
for _ in range(n_steps):
    f_d = hev.dense_step(f_d, rhs_d, dt, "rk2")
ex = hev.spectral_reference(f0, n_steps * dt, cx, cy, kappa, dx)
fd_err = np.abs(f_d - ex).max() / np.abs(ex).max()
print(f"     dense FD vs spectral = {fd_err:.3e}   <- the floor no TT run can beat")

print("\n=== 5. convergence: TT error must vanish as bd -> full rank ===")
prev = np.inf
for bd in [4, 8, 16, 32, 64]:
    h0 = ho.field_to_hybrid(F0, L, N, bd_first=bd, bd_remaining=bd)
    f_start = ho.hybrid_to_field(h0, L).numpy()
    trunc = hev.Truncation(bd_first=bd, bd_remaining=bd)
    _, hist, _ = hev.evolve(h0, L, N, op, dt, 50, trunc, scheme="rk2",
                            dense_f0=f_start,
                            dense_rhs=hev.dense_rhs_adv_diff(cx, cy, kappa, dx),
                            every=50)
    e = hist.err_vs_dense[-1]
    mono = e <= prev + 1e-13
    if not mono:
        FAILURES.append(f"error not monotone in bd at bd={bd}")
    print(f"  {'ok ' if mono else 'BAD'} bd={bd:>2}: err vs dense-FD = {e:.3e}, "
          f"variance lost to truncation = {trunc.lost_norm2:.3e}")
    prev = e

print("\n=== 6. steppers agree with their dense counterparts ===")
for scheme in ["euler", "rk2", "ssprk3", "rk4"]:
    h0 = ho.field_to_hybrid(F0, L, N, bd_first=2**N, bd_remaining=2**L, eps=1e-15)
    f_start = ho.hybrid_to_field(h0, L).numpy()
    trunc = hev.Truncation(bd_first=2**N, bd_remaining=2**L, eps=1e-13)
    _, hist, dense = hev.evolve(h0, L, N, op, dt, 20, trunc, scheme=scheme,
                                dense_f0=f_start,
                                dense_rhs=hev.dense_rhs_adv_diff(cx, cy, kappa, dx),
                                every=20)
    check(f"{scheme} matches dense {scheme}", hist.err_vs_dense[-1], 1e-9)

print("\n" + "=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    sys.exit(1)
print("all checks passed")
