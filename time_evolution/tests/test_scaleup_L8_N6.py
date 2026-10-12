"""Everything validated so far, re-run at L=8, N=6 through the fused-head path.

Run:  python tests/test_scaleup_L8_N6.py

The head is treated as a single explicit 2^N x 2^N operator built once
(hybrid_ops.fuse_head_operator) rather than un-fused into N binary legs on every
application.  At L=8 the 256x256 dense field still fits, so every result is
checked against a dense reference rather than only against itself.
"""
import os
import sys
import time
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
    print(f"  {'PASS' if good else '**FAIL**':10s} {name:52s} err={err:.3e} (tol {tol:.0e})")


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


L, N = 8, 6
n = 2**L
FULL = 2**N
rng = np.random.default_rng(31)

x, y = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
blob = np.exp(-(((x - 90) ** 2 + (y - 150) ** 2) / 500.0))
kk = np.fft.fftfreq(n) * n
KX, KY = np.meshgrid(kk, kk, indexing="xy")
K = np.sqrt(KX**2 + KY**2)
K[0, 0] = 1.0
noise = np.real(np.fft.ifft2((rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * K**-1.5))
f0 = blob + 0.5 * noise / noise.std()
F0 = torch.from_numpy(f0)

print(f"L={L} ({n}x{n} = {n*n} cells), N={N}")
print(f"  head physical dim 2^{N} = {FULL};  tail sites {2*L-N};  total legs {2*L}")
print(f"  coarse blocks {2**(N//2)}x{2**(N//2)};  seam period 2^(L-N/2) = {ho.seam_period(L, N)}")

# ==========================================================================
print("\n=== 1. fused head operators: inventory ===")
OPS = {
    "d/dx":        hop.derivative_mpo(L, hop.AXIS_X, hop.D1_CENTRAL_2),
    "d/dy":        hop.derivative_mpo(L, hop.AXIS_Y, hop.D1_CENTRAL_2),
    "laplacian":   hop.laplacian_mpo(L),
    "translate x": hop.translation_mpo(L, hop.AXIS_X, 1),
    "seam smooth": hop.seam_smoother_mpo(L, N),
}
print(f"  {'operator':<13} {'MPO bond':>9} {'blocks':>7} {'dense KB':>9} {'nnz':>7} {'fill':>7}")
HEADS = {}
for k, op in OPS.items():
    Hd = ho.fuse_head_operator(op, N, sparse=False)
    Hs = ho.fuse_head_operator(op, N, sparse=True)
    HEADS[k] = Hd
    nnz = ho.head_operator_nnz(Hs)
    tot = len(Hd) * FULL * FULL
    print(f"  {k:<13} {op.max_rank:>9} {len(Hd):>7} {len(Hd)*FULL*FULL*8/1024:>9.0f} "
          f"{nnz:>7} {100*nnz/tot:>6.2f}%")

# ==========================================================================
print("\n=== 2. fused head == un-fused head == dense reference ===")
h_exact = ho.field_to_hybrid(F0, L, N, bd_first=FULL, bd_remaining=2**L, eps=1e-13)
check("lossless hybrid round trip", rel(ho.hybrid_to_field(h_exact, L).numpy(), f0), 1e-11)


def dense_1d(coef):
    M = np.zeros((n, n))
    for i in range(n):
        for k, v in coef.items():
            if v:
                M[i, (i + k) % n] += v
    return M


D1 = dense_1d(hop.D1_CENTRAL_2)
D2 = dense_1d(hop.D2_CENTRAL_2)
REF = {
    "d/dx":        f0 @ D1.T,
    "d/dy":        D1 @ f0,
    "laplacian":   f0 @ D2.T + D2 @ f0,
    "translate x": np.roll(f0, 1, axis=hop.AXIS_X),
    "seam smooth": hop.seam_smooth_dense(F0, L, N).numpy(),
}
for k, op in OPS.items():
    un = ho.hybrid_to_field(ho.hybrid_apply(op, h_exact, N), L).numpy()
    fu = ho.hybrid_to_field(ho.hybrid_apply_fused(op, h_exact, N, head_op=HEADS[k]), L).numpy()
    check(f"{k}: fused vs un-fused", rel(fu, un), 1e-13)
    check(f"{k}: fused vs dense", rel(fu, REF[k]), 1e-11)

print("\n  -- pointwise evaluation without contracting (needed above L~10) --")
ri, ci = rng.integers(0, n, 2000), rng.integers(0, n, 2000)
check("hybrid_eval_at vs dense", rel(ho.hybrid_eval_at(h_exact, L, ri, ci).numpy(), f0[ri, ci]), 1e-11)

# ==========================================================================
print("\n=== 3. timing: fused vs un-fused head, per application ===")
h_t = ho.field_to_hybrid(F0, L, N, bd_first=32, bd_remaining=32)
reps = 200
print(f"  {'operator':<13} {'un-fused':>11} {'fused dense':>12} {'fused sparse':>13} {'speedup':>9}")
for k in ["d/dx", "laplacian", "seam smooth"]:
    op = OPS[k]
    Hd, Hs = HEADS[k], ho.fuse_head_operator(op, N, sparse=True)
    t0 = time.perf_counter()
    for _ in range(reps):
        ho.hybrid_apply(op, h_t, N)
    t_un = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        ho.hybrid_apply_fused(op, h_t, N, head_op=Hd)
    t_fd = (time.perf_counter() - t0) / reps
    t0 = time.perf_counter()
    for _ in range(reps):
        ho.hybrid_apply_fused(op, h_t, N, head_op=Hs)
    t_fs = (time.perf_counter() - t0) / reps
    print(f"  {k:<13} {t_un*1e3:>9.3f}ms {t_fd*1e3:>10.3f}ms {t_fs*1e3:>11.3f}ms "
          f"{t_un/min(t_fd, t_fs):>8.2f}x")

# ==========================================================================
print("\n=== 4. rank behaviour under translation (adaptive eps, no cap) ===")
T = OPS["translate x"]
period = ho.seam_period(L, N)
# A SMOOTH field is required here: the multiscale f0 used elsewhere is full rank
# at every tolerance, so rank growth would be clipped by the chain geometry
# (max bond 2^L at the middle of a 2L-leg chain) and nothing would be visible.
f_sm = (np.exp(-(((x - 90.) ** 2 + (y - 150.) ** 2) / 900.))
        + 0.3 * np.sin(4 * np.pi * x / n) * np.cos(2 * np.pi * y / n))
F_sm = torch.from_numpy(f_sm)
shifts = [0, 1, 2, 8, 16, 31, 32, 33, 64]
print(f"  coarse-block width = {period}; realignment expected at multiples of {period}")
print(f"  {'eps':>7} " + " ".join(f"{'i='+str(i):>5}" for i in shifts))
for eps in [1e-6, 1e-8, 1e-10]:
    rk = [max(ho.bond_dims(ho.field_to_hybrid(
        torch.from_numpy(np.roll(f_sm, i, axis=hop.AXIS_X)), L, N,
        bd_first=FULL, bd_remaining=2**L, eps=eps))) for i in shifts]
    print(f"  {eps:>7.0e} " + " ".join(f"{r:>5d}" for r in rk) + "   (intrinsic)")
    if eps == 1e-8:
        base, peak = rk[0], max(rk)
        aligned = [rk[i] for i, sh in enumerate(shifts) if sh % period == 0]
        if set(aligned) != {base}:
            FAILURES.append("rank does not return to baseline on aligned shifts")
        print(f"          -> baseline {base}, peak {peak} (+{100*(peak-base)/base:.0f}%), "
              f"aligned shifts all {sorted(set(aligned))}")
for eps in [1e-6, 1e-8, 1e-10]:
    h = ho.field_to_hybrid(F_sm, L, N, bd_first=FULL, bd_remaining=2**L, eps=eps)
    rk = {0: max(ho.bond_dims(h))}
    for i in range(1, 65):
        h = ho.hybrid_compress(ho.hybrid_apply_fused(T, h, N, head_op=HEADS["translate x"]),
                               bd_first=FULL, bd_remaining=2**L, eps=eps)
        if i in shifts:
            rk[i] = max(ho.bond_dims(h))
    err = rel(ho.hybrid_to_field(h, L).numpy(), np.roll(f_sm, 64, axis=hop.AXIS_X))
    print(f"  {eps:>7.0e} " + " ".join(f"{rk[i]:>5d}" for i in shifts)
          + f"   (evolved, err@64 = {err:.1e})")

print("\n  -- fraction of cells the seam smoother touches, by configuration --")
for LL, NN in [(6, 6), (8, 6), (8, 8), (12, 12)]:
    per = 2 ** (LL - NN // 2)
    print(f"     L={LL:>2} N={NN:>2}: period {per:>4} -> {100*(2/per - 1/per**2):5.2f}% of cells")

# ==========================================================================
print("\n=== 5. truncated advection: 256 steps ===")
BD = 16
h_ic = ho.field_to_hybrid(F0, L, N, bd_first=BD, bd_remaining=BD)
f_ic = ho.hybrid_to_field(h_ic, L).numpy()
print(f"  bd={BD}. Compressing the IC alone already costs {rel(f_ic, f0):.3e}.")
Hseam = HEADS["seam smooth"]
Htr = HEADS["translate x"]
configs = [("seam off", None, 1, False), ("seam every step", "S", 1, False),
           ("seam every 32", "S", 32, False), ("seam off + mass", None, 1, True)]
res = {}
for tag, useS, sev, mass in configs:
    h = ho.field_to_hybrid(F0, L, N, bd_first=BD, bd_remaining=BD)
    trunc = hev.Truncation(bd_first=BD, bd_remaining=BD)
    mean0 = ho.hybrid_sum(h) / 4**L
    hist = hev.History()
    hev.record(hist, 0.0, h, L, N, f_ic, None)
    for i in range(1, 257):
        h = trunc(ho.hybrid_apply_fused(T, h, N, head_op=Htr))
        if useS and i % sev == 0:
            h = trunc(ho.hybrid_apply_fused(OPS["seam smooth"], h, N, head_op=Hseam))
        if mass:
            h = ho.project_mass(h, L, N, mean0)
        if i % 64 == 0:
            hev.record(hist, float(i), h, L, N, np.roll(f_ic, i, axis=hop.AXIS_X), None)
    res[tag] = (hist, trunc)

print(f"  {'configuration':<18} {'step':>5} {'err':>11} {'variance':>10} {'seam':>6} {'mass drift':>11}")
for tag, _, _, _ in configs:
    hist, _ = res[tag]
    for j in range(1, len(hist.t)):
        md = abs(hist.mass[j] - hist.mass[0]) / max(abs(hist.mass[0]), 1e-12)
        print(f"  {tag if j == 1 else '':<18} {int(hist.t[j]):>5} {hist.err_vs_dense[j]:>11.4e} "
              f"{hist.variance[j]:>10.5f} {hist.seam[j]:>6.2f} {md:>11.2e}")
v0 = res["seam off"][0].variance[0]
print()
for tag, _, _, _ in configs:
    hist, trunc = res[tag]
    print(f"  {tag:<18} variance retained {100*hist.variance[-1]/v0:5.1f}%   "
          f"seam {hist.seam[0]:.2f} -> {hist.seam[-1]:.2f}")
hoff = res["seam off"][0]
g = hoff.err_vs_dense[-1] / max(hoff.err_vs_dense[1], 1e-30)
print(f"\n  error growth, step 64 -> 256 (4x more steps): {g:.2f}x "
      f"({'saturating' if g < 2 else 'compounding'})")
if g > 4.0:
    FAILURES.append("error growth faster than linear")
mf = max(abs(m - res['seam off + mass'][0].mass[0]) for m in res['seam off + mass'][0].mass)
check("project_mass restores mass", mf / max(abs(hoff.mass[0]), 1e-12), 1e-12)

# ==========================================================================
print("\n=== 6. advection-diffusion vs dense FD and vs spectral ===")
cx, cy, kappa, dx = 0.7, -0.4, 0.05, 1.0
dt = hev.stable_dt(cx, cy, kappa, dx)
nst = 200
op = hop.advection_diffusion_mpo(L, cx=cx, cy=cy, kappa=kappa, dx=dx)
print(f"  RHS MPO bond {op.max_rank}, dt={dt:.3f}, {nst} steps, "
      f"CFL={max(abs(cx),abs(cy))*dt/dx:.2f}, diff num={kappa*dt/dx**2:.3f}")
prev = np.inf
for bd in [8, 16, 32, 64]:
    h0 = ho.field_to_hybrid(F0, L, N, bd_first=bd, bd_remaining=bd)
    fs = ho.hybrid_to_field(h0, L).numpy()
    trunc = hev.Truncation(bd_first=bd, bd_remaining=bd)
    _, hist, _ = hev.evolve(h0, L, N, op, dt, nst, trunc, scheme="rk2", fused=True,
                            dense_f0=fs, dense_rhs=hev.dense_rhs_adv_diff(cx, cy, kappa, dx),
                            exact_fn=lambda t: hev.spectral_reference(fs, t, cx, cy, kappa, dx),
                            every=nst)
    e = hist.err_vs_dense[-1]
    if e > prev + 1e-13:
        FAILURES.append(f"error not monotone in bd at bd={bd}")
    print(f"  bd={bd:>2}: TT-only err = {e:.3e}   total err = {hist.err_vs_exact[-1]:.3e}   "
          f"seam {hist.seam[0]:.2f} -> {hist.seam[-1]:.2f}")
    prev = e
fd = f0.copy()
rhs_d = hev.dense_rhs_adv_diff(cx, cy, kappa, dx)
for _ in range(nst):
    fd = hev.dense_step(fd, rhs_d, dt, "rk2")
ex = hev.spectral_reference(f0, nst * dt, cx, cy, kappa, dx)
print(f"  dense FD vs spectral = {rel(fd, ex):.3e}   <- floor no TT run can beat")

print("\n=== 7. fused and un-fused give identical evolution ===")
for scheme in ["rk2", "ssprk3"]:
    outs = []
    for fused in [False, True]:
        h0 = ho.field_to_hybrid(F0, L, N, bd_first=24, bd_remaining=24)
        trunc = hev.Truncation(bd_first=24, bd_remaining=24)
        hh, _, _ = hev.evolve(h0, L, N, op, dt, 25, trunc, scheme=scheme, fused=fused, every=25)
        outs.append(ho.hybrid_to_field(hh, L).numpy())
    check(f"{scheme}: 25 steps fused == un-fused", rel(outs[1], outs[0]), 1e-12)

print("\n" + "=" * 72)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    sys.exit(1)
print("all checks passed")
