"""Kraichnan model: plane-wave TTs, the sampler, the hybrid interval, the lattice benchmark.

Run:  python tests/test_kraichnan.py

1. A plane wave is an exact rank-2 hybrid TT for every head size, including N=0.
2. The analytic velocity equals the solver's discrete curl of the streamfunction
   (derivative MPOs), and is divergence-free on the grid.
3. The sampler's amplitudes reproduce the target covariances exactly, with one
   Kubo number for every mode.
4. A hybrid interval (substeps plus forcing increment), run without effective
   truncation, matches the dense interval.
5. The closed lattice equation for C(k) equals the phase-averaged double
   commutator (tau/2) E[A,[A,M]] of the actual dense transport operator, built
   column by column on a 16x16 grid: diagonal entries match and off-diagonal
   ones vanish. This is the derivation in kraichnan.py, checked including
   aliasing and the discrete symbols.
6. The lattice equation conserves variance in the transfer term, and its
   stationary solve converges.
7. The increment statistics (structure functions, PDFs) are correct, and a
   lossless hybrid run reproduces them and records every output key; the same
   with the budget policy.
8. Budget-consistent compression (hybrid_evolve.PhysicalBudget): exact inner
   products and variance production; skew-form transport produces no variance;
   the SSPRK3 quadrature of the budget differs from the dense scheme's own norm
   change at O(dt^4) only; a compressed step lands exactly on the budget; a
   lossless budgeted step departs from the plain dense step by that O(dt^4) term
   alone and equals the budgeted dense step, also with a steady source of
   non-zero mean, whose mass is then added exactly; compressed updates land
   exactly on both the mass and the variance budget;
   the forcing update keeps its exact norm.
"""
import os
import sys
import warnings

_PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # time_evolution/
sys.path.insert(0, _PKG)
sys.path.insert(0, os.path.dirname(_PKG))

import numpy as np
import torch

import config as cfg

sys.path.append(cfg.TENSNET_SRC)

import hybrid_evolve as hev  # noqa: E402
import hybrid_ops as ho  # noqa: E402
import kraichnan as kr  # noqa: E402
import run_decaying_cellular as rdc  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)
FAILURES = []


def check(name, err, tol):
    good = bool(err <= tol)
    if not good:
        FAILURES.append(name)
    print(f"  {'PASS' if good else '**FAIL**':10s} {name:58s} err={err:.3e} (tol {tol:.0e})")


rng = np.random.default_rng(11)

# ==========================================================================
print("=== 1. plane waves are exact rank-2 hybrid TTs ===")
L = 5
n, dx, XX, YY = rdc.grid(L)
for N in (0, 2, 4, 6):
    for (qx, qy) in ((1, 0), (0, 3), (-2, 5), (7, -4)):
        ph, amp = rng.uniform(0, 2 * np.pi), rng.uniform(0.5, 2)
        h = kr.plane_wave_hybrid(L, N, qx, qy, ph, amp)
        ho.validate_hybrid(h, L, N)
        got = ho.hybrid_to_field(h, L).numpy()
        ref = amp * np.cos(qx * XX + qy * YY + ph)
        check(f"N={N} q=({qx},{qy}) field", np.abs(got - ref).max(), 1e-12)
        check(f"N={N} q=({qx},{qy}) bond <= 2", max(ho.bond_dims(h)) - 2, 0)

# ==========================================================================
print("\n=== 2. velocity: analytic = discrete curl, divergence-free ===")
L, N = 5, 4
n, dx, XX, YY = rdc.grid(L)
model = kr.Kraichnan(L, xi=4 / 3, qmax=4)
for i in rng.choice(len(model.q), 4, replace=False):
    ph = rng.uniform(0, 2 * np.pi)
    uxd, uyd = model.velocity_dense(i, ph, XX, YY)
    ux, uy = model.velocity_hybrid(i, ph, N)
    qx, qy = model.q[i]
    psi = kr.plane_wave_hybrid(L, N, qx, qy, ph, model.psi_amp[i])
    vx, vy = hev.velocity_from_streamfunction(psi, L, N, dx)
    for nm, a, b in (("ux", ux, vx), ("uy", uy, vy)):
        A, B = ho.hybrid_to_field(a, L).numpy(), ho.hybrid_to_field(b, L).numpy()
        check(f"q=({qx},{qy}) {nm}: hybrid = curl MPO", np.abs(A - B).max() / model.amp[i], 1e-12)
    check(f"q=({qx},{qy}) dense = hybrid",
          np.abs(ho.hybrid_to_field(ux, L).numpy() - uxd).max() / model.amp[i], 1e-12)
    div = (hev._apply_stencil_dense(uxd, rdc.hop.D1_CENTRAL_2, 1)
           + hev._apply_stencil_dense(uyd, rdc.hop.D1_CENTRAL_2, 0)) / dx
    check(f"q=({qx},{qy}) discrete divergence", np.abs(div).max() / model.amp[i] * dx, 1e-13)

# ==========================================================================
print("\n=== 3. sampler amplitudes ===")
for xi, qmax in ((2.0, np.sqrt(2)), (4 / 3, 8)):
    m = kr.Kraichnan(6, xi=xi, qmax=qmax, ku=0.4)
    s2 = (m.s ** 2).sum(1)
    check(f"xi={xi:.2f}: tau p a^2 / 2 = w", np.abs(m.tau * m.p * m.amp ** 2 / 2 - m.w).max()
          / m.w.max(), 1e-12)
    check(f"xi={xi:.2f}: Kubo number uniform", np.abs(m.amp * m.tau * np.sqrt(s2) - m.ku).max(), 1e-12)
    check(f"xi={xi:.2f}: forcing p c^2 / (2 tau) = phi",
          np.abs(m.pf * m.famp ** 2 / (2 * m.tau) - m.phi).max(), 1e-12)
    check(f"xi={xi:.2f}: Gamma(k_f) = 1", abs(m.gamma(kr.KF) - 1), 1e-12)
    qq = {tuple(q) for q in m.q}
    check(f"xi={xi:.2f}: half-plane holds no +-q pair",
          sum((-qx, -qy) in qq for qx, qy in qq), 0)
    draws = np.array([m.draw(rng)[0] for _ in range(200000)])
    freq = np.bincount(draws, minlength=len(m.q)) / len(draws)
    check(f"xi={xi:.2f}: draw frequencies follow p (200k draws)",
          np.abs(freq - m.p).max() / np.sqrt(m.p.max() / len(draws)), 5.0)

# ==========================================================================
print("\n=== 4. hybrid interval = dense interval ===")
L, N = 5, 4
n, dx, XX, YY = rdc.grid(L)
model = kr.Kraichnan(L, xi=4 / 3, qmax=4, ku=0.5)
kappa = 0.02
f0 = rdc.initial_field(L, kmax=3)
dense = f0.copy()
h = ho.field_to_hybrid(torch.from_numpy(f0), L, N, bd_first=None, eps=1e-15)
trunc = hev.Truncation(eps=1e-14)
ops = hev.transport_operators(L, dx, kappa)
draw_rng = np.random.default_rng(5)
for _ in range(3):
    iv, ph, jf, phf = model.draw(draw_rng)
    a = model.amp[iv]
    m_sub = int(np.ceil(model.tau / hev.stable_dt(a, a, kappa, dx, cfl=0.5) - 1e-9))
    dt = model.tau / m_sub
    uxd, uyd = model.velocity_dense(iv, ph, XX, YY)
    rhs_d = hev.dense_variable_rhs(uxd, uyd, dx, kappa=kappa)
    ux, uy = model.velocity_hybrid(iv, ph, N)
    rhs_h = hev.variable_rhs(ux, uy, L, N, dx, kappa=kappa, trunc=trunc, ops=ops)
    for _ in range(m_sub):
        dense = hev.dense_step(dense, rhs_d, dt, "ssprk3")
        h = hev.step_ssprk3(h, rhs_h, dt, trunc)
    dense = dense + model.forcing_dense(jf, phf, XX, YY)
    h = trunc(ho.hybrid_add(h, model.forcing_hybrid(jf, phf, N)))
check("3 intervals, hybrid vs dense (eps=1e-14)",
      np.abs(ho.hybrid_to_field(h, L).numpy() - dense).max() / np.abs(dense).max(), 1e-10)

# ==========================================================================
print("\n=== 5. lattice equation = phase-averaged double commutator (16x16) ===")
L = 4
n, dx, XX, YY = rdc.grid(L)
model = kr.Kraichnan(L, xi=4 / 3, qmax=2, ku=0.5)
op = kr.LatticeOperator(model, kappa=0.0)
# unitary 2-D DFT on the row-major flattened field
F1 = np.fft.fft(np.eye(n), axis=0) / np.sqrt(n)
F = np.kron(F1, F1)
C = rng.uniform(0.5, 2.0, (n, n))
C = 0.5 * (C + C[(-np.arange(n)) % n][:, (-np.arange(n)) % n])   # C(k) = C(-k)
C[0, 0] = 0.0
M = np.diag(C.ravel()).astype(complex)
R = np.zeros((n * n, n * n), dtype=complex)
for i in range(len(model.q)):
    for ph in (0.0, np.pi / 2, np.pi, 3 * np.pi / 2):
        uxd, uyd = model.velocity_dense(i, ph, XX, YY)
        rhs = hev.dense_variable_rhs(uxd, uyd, dx, kappa=0.0)
        A = -np.stack([rhs(e.reshape(n, n)).ravel() for e in np.eye(n * n)], 1)
        Ah = F @ A @ F.conj().T
        comm = Ah @ M - M @ Ah
        R += model.p[i] * model.tau / 2 * (Ah @ comm - comm @ Ah) / 4
pred = op.gain(C) - op.gain1 * C
scale = np.abs(pred).max()
check("diagonal: lattice formula vs brute force", np.abs(np.real(np.diag(R)).reshape(n, n)
                                                         - pred).max() / scale, 1e-11)
check("off-diagonal entries vanish", np.abs(R - np.diag(np.diag(R))).max() / scale, 1e-11)
check("A is anti-Hermitian in Fourier space", np.abs(Ah + Ah.conj().T).max(), 1e-11)

# ==========================================================================
print("\n=== 6. lattice transfer conserves variance; stationary solve ===")
model = kr.Kraichnan(6, xi=4 / 3, qmax=8, ku=0.5)
op = kr.LatticeOperator(model, kappa=0.0)
C = rng.uniform(0, 1, (64, 64))
C = 0.5 * (C + C[(-np.arange(64)) % 64][:, (-np.arange(64)) % 64])
t = op.gain(C) - op.gain1 * C
check("sum_k transfer(C) = 0", abs(t.sum()) / np.abs(t).sum(), 1e-12)
kappa = model.default_kappa()
Cs, info = kr.lattice_spectrum(model, kappa)
check("stationary solve residual", info["residual"], 1e-9)
check("stationary spectrum non-negative", max(-Cs.min(), 0.0) / Cs.max(), 1e-10)
# the variance budget: injection Phi(0) = dissipation 2 kappa sum lam C
_, _, _, _, lam = kr._lattice(64, model.dx)
check("budget: injection = 2 kappa <lam C>",
      abs(2 * kappa * (lam * Cs).sum() - model.injection) / model.injection, 1e-8)
print(f"  ({info['iterations']} GMRES iterations, variance {Cs.sum():.4f})")

# ==========================================================================
print("\n=== 7. increment statistics; end-to-end hybrid run ===")
import tempfile  # noqa: E402

import run_kraichnan as rk  # noqa: E402

f = rdc.initial_field(6, kmax=12)
n = f.shape[0]
r_sf = rk.separations(n)
st = rk.increment_stats(f, r_sf)
# S_2(r) = 2 (<f^2> - C(r)), with C the periodic autocorrelation, pooled over x and y
C = np.real(np.fft.ifft2(np.abs(np.fft.fft2(f)) ** 2)) / f.size
s2 = np.array([2 * (C[0, 0] - 0.5 * (C[0, r] + C[r, 0])) for r in r_sf])
check("S_2(r) = 2 (C(0) - C(r)) via FFT", np.abs(st["sf"][0] - s2).max() / s2.max(), 1e-12)
check("histograms hold every increment", np.abs(st["pdf"].sum(1) - 2 * n * n).max(), 0)
d = rk.increments(f, 4)
check("S_4(r=4) by brute force", abs(st["sf"][1, list(r_sf).index(4)] - (d ** 4).mean())
      / (d ** 4).mean(), 1e-12)
with tempfile.TemporaryDirectory() as tmp:
    rk.RESULTS = tmp
    np.save(os.path.join(tmp, "init.npy"), f)
    path = rk.run(6, 4 / 3, 8.0, 0.5, None, 1.0, 0.5, 0.1, 4, 64, False, 3,
                  os.path.join(tmp, "init.npy"), None, 5, 2, 0.0, 0, "t7")
    out = np.load(path)
    # bd = 64 is the geometric bound at L = 6, so the hybrid run is exact
    check("lossless hybrid run: err_l2", out["err_l2"].max(), 1e-9)
    check("lossless hybrid run: S_p(TT) = S_p(ref)",
          np.abs(out["tt_sf"] - out["ref_sf"]).max() / out["ref_sf"].max(), 1e-8)
    check("ref_snaps saved alongside snaps", abs(out["ref_snaps"].shape[0] - out["snaps"].shape[0]), 0)
    check("wall is a scalar total", out["wall"].ndim, 0)
    # budget policy (hybrid_evolve.PhysicalBudget)
    init = os.path.join(tmp, "init.npy")
    out = np.load(rk.run(6, 4 / 3, 8.0, 0.5, None, 1.0, 0.5, 0.1, 4, 64, False, 3,
                         init, None, 5, 2, 0.0, 0, "t7b", policy="budget"))
    print(f"  (lossless budgeted run: max |s-1| = {out['budget_dev_max'].max():.2e})")
    check("lossless budgeted run = budgeted dense twin", out["err_l2"].max(), 1e-9)
    out = np.load(rk.run(6, 4 / 3, 8.0, 0.5, None, 1.0, 0.5, 0.1, 4, 6, False, 3,
                         init, None, 5, 2, 0.0, 0, "t7b6", policy="budget"))
    ms = lambda w: out[f"{w}_variance"] + out[f"{w}_mass"] ** 2  # noqa: E731
    check("chi=6, budget: |theta_TT| = |theta| at t = 0", abs(ms("tt")[0] / ms("ref")[0] - 1),
          1e-10)
    check("chi=6, budget: initial error = renormalised control",
          abs(out["err_l2"][0] - out["repr_err_l2"][0]) / out["repr_err_l2"][0], 1e-8)
    check("chi=6, budget: mass exact at every sample",
          np.abs(out["tt_mass"] - out["ref_mass"]).max() / np.sqrt(out["ref_variance"].min()), 1e-12)
    check("chi=6, budget: state bond = chi + 1 (the uniform mode)", abs(out["tt_bonds"].max() - 7), 0)
    check("chi=6: run records policy=budget", float(str(out["policy"]) != "budget"), 0)

# ==========================================================================
print("\n=== 8. budget-consistent compression ===")
L, N = 5, 4
n, dx, XX, YY = rdc.grid(L)
hop = rdc.hop
model = kr.Kraichnan(L, xi=4 / 3, qmax=4, ku=0.5)
kappa = 0.02
ops = hev.transport_operators(L, dx, kappa)


def lap_dense(f):
    return (hev._apply_stencil_dense(f, hop.D2_CENTRAL_2, hop.AXIS_X)
            + hev._apply_stencil_dense(f, hop.D2_CENTRAL_2, hop.AXIS_Y)) / dx ** 2


f = rdc.initial_field(L, kmax=6)
g = rdc.initial_field(L, kmax=3) ** 2
hf = ho.field_to_hybrid(torch.from_numpy(f), L, N, bd_first=None, eps=1e-15)
hg = ho.field_to_hybrid(torch.from_numpy(g), L, N, bd_first=None, eps=1e-15)
check("hybrid_inner = dense dot", abs(ho.hybrid_inner(hf, hg) - (f * g).sum()) / abs((f * g).sum()),
      1e-12)
bud = hev.PhysicalBudget(L, N, kappa, lap=ops[2])
p_dense = 2 * kappa * (f * lap_dense(f)).sum()
check("production = 2 kappa <f, lap f> (dense)", abs(bud.production(hf) - p_dense) / abs(p_dense),
      1e-11)
iv = int(np.argmax(model.p))
uxd, uyd = model.velocity_dense(iv, 0.3, XX, YY)
adv = hev.dense_variable_rhs(uxd, uyd, dx, kappa=0.0, form="skew")
r = adv(f)
check("skew transport produces no variance: <f, A_adv f>", abs((f * r).sum())
      / (np.linalg.norm(f) * np.linalg.norm(r)), 1e-13)

# the SSPRK3 quadrature of the budget against the dense scheme's own norm change
rhs_d = hev.dense_variable_rhs(uxd, uyd, dx, kappa=kappa, form="skew")
P_d = lambda x: 2 * kappa * (x * lap_dense(x)).sum()  # noqa: E731


def mismatch(dt):
    u1 = f + dt * rhs_d(f)
    u2 = 0.75 * f + 0.25 * (u1 + dt * rhs_d(u1))
    u3 = f / 3 + 2 / 3 * (u2 + dt * rhs_d(u2))
    target = (f ** 2).sum() + dt * (P_d(f) / 6 + P_d(u1) / 6 + 2 * P_d(u2) / 3)
    return abs((u3 ** 2).sum() - target) / (f ** 2).sum()


a = float(model.amp[iv])
dt0 = hev.stable_dt(a, a, kappa, dx, cfl=0.5)
m1, m2, m4 = mismatch(dt0), mismatch(dt0 / 2), mismatch(dt0 / 4)
print(f"  (dense |u|^2 - budget per step: {m1:.2e}, {m2:.2e}, {m4:.2e} at dt, dt/2, dt/4)")
check("dense scheme vs budget quadrature: O(dt^4) (ratio dt/2 : dt/4 ~ 1/16)",
      abs(np.log2(m2 / m4) - 4.0), 0.6)

# compressed budgeted step lands exactly on the budget
class Rec(hev.PhysicalBudget):
    def project(self, h, target_sum, target_norm2):
        self.last, self.last_sum = target_norm2, target_sum
        return super().project(h, target_sum, target_norm2)


ux, uy = model.velocity_hybrid(iv, 0.3, N)
trunc6 = hev.Truncation(bd_first=6, bd_remaining=6)
rhs6 = hev.variable_rhs(ux, uy, L, N, dx, kappa=kappa, form="skew", trunc=trunc6, ops=ops)
rec = Rec(L, N, kappa, lap=ops[2])
h6 = trunc6(hf)
out6 = hev.step_ssprk3_budget(h6, rhs6, dt0, trunc6, rec)
check("chi=6 step: |theta|^2 = integrated budget", abs(ho.hybrid_norm(out6) ** 2 / rec.last - 1),
      1e-12)
check("chi=6 step: mass = integrated budget (conserved)",
      abs(ho.hybrid_sum(out6) - ho.hybrid_sum(h6)) / (n * ho.hybrid_norm(out6)), 1e-13)
inc = model.forcing_hybrid(int(np.argmax(model.pf)), 1.1, N)
o = rec.update_with(out6, inc, trunc6)
exact = ho.hybrid_to_field(ho.hybrid_add(out6, inc), L).numpy()
check("forcing update keeps its exact norm", abs(ho.hybrid_norm(o) / np.linalg.norm(exact) - 1), 1e-12)
check("forcing update keeps its exact mass", abs(ho.hybrid_sum(o) - exact.sum()) / (n * np.linalg.norm(exact)),
      1e-13)

# lossless budgeted step = dense step up to the O(dt^4) term of the scheme
cap = 2 ** L
trunc_x = hev.Truncation(bd_first=cap, bd_remaining=cap)
rhs_x = hev.variable_rhs(ux, uy, L, N, dx, kappa=kappa, form="skew", trunc=trunc_x, ops=ops)
devs = []
for dt in (dt0 / 2, dt0 / 4):
    b = hev.PhysicalBudget(L, N, kappa, lap=ops[2])
    hev.step_ssprk3_budget(hf, rhs_x, dt, trunc_x, b)
    devs.append(b.dev_max)
print(f"  (lossless |s-1|: {devs[0]:.2e}, {devs[1]:.2e} at dt/2, dt/4)")
check("lossless step: |s-1| = O(dt^4), the scheme's own term", abs(np.log2(devs[0] / devs[1]) - 4.0),
      0.6)
b = hev.PhysicalBudget(L, N, kappa, lap=ops[2])
hx = hev.step_ssprk3_budget(hf, rhs_x, dt0, trunc_x, b)
fx = hev.dense_step_ssprk3_budget(f, rhs_d, dt0, kappa, dx)
check("lossless budgeted step = dense budgeted step",
      np.abs(ho.hybrid_to_field(hx, L).numpy() - fx).max() / np.abs(fx).max(), 1e-12)
# steady source with non-zero mean: its mass enters at rate sum(g)
rhs_xg = hev.variable_rhs(ux, uy, L, N, dx, kappa=kappa, form="skew", forcing=hg,
                          trunc=trunc_x, ops=ops)
rhs_dg = hev.dense_variable_rhs(uxd, uyd, dx, kappa=kappa, form="skew", forcing=g)
bg = hev.PhysicalBudget(L, N, kappa, lap=ops[2], forcing=hg)
hxg = hev.step_ssprk3_budget(hf, rhs_xg, dt0, trunc_x, bg)
fxg = hev.dense_step_ssprk3_budget(f, rhs_dg, dt0, kappa, dx, forcing=g)
check("with a steady source: lossless budgeted step = dense",
      np.abs(ho.hybrid_to_field(hxg, L).numpy() - fxg).max() / np.abs(fxg).max(), 1e-12)
check("with a steady source: mass grows by dt * sum(f)",
      abs(fxg.sum() - f.sum() - dt0 * g.sum()) / abs(dt0 * g.sum()), 1e-10)

print("\n" + "=" * 70)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    sys.exit(1)
print("all checks passed")
