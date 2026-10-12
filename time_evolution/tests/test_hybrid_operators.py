"""Validation of hybrid_operators against dense references.

Run:  python tests/test_hybrid_operators.py
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

import hybrid_operators as hop  # noqa: E402
import hybrid_ops as ho  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float64)

FAILURES = []


def check(name, err, tol):
    good = bool(err <= tol)
    if not good:
        FAILURES.append(name)
    print(f"  {'PASS' if good else '**FAIL**':10s} {name:54s} err={err:.3e} (tol {tol:.0e})")


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-300))


L, N = 5, 4
n = 2**L
rng = np.random.default_rng(11)
xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
f = (np.exp(-((xx - 9) ** 2 + (yy - 21) ** 2) / 18.0)
     + 0.5 * np.sin(2 * np.pi * xx / n) * np.cos(6 * np.pi * yy / n)
     + 0.2 * rng.standard_normal((n, n)))
F = torch.from_numpy(f)
hyb = ho.field_to_hybrid(F, L, N, bd_first=2**N, bd_remaining=2**L, eps=1e-15)


def dense_1d(coef):
    M = np.zeros((n, n))
    for i in range(n):
        for k, v in coef.items():
            if v:
                M[i, (i + k) % n] += v
    return M


def apply_dense(coef, axis):
    D = dense_1d(coef)
    return D @ f if axis == hop.AXIS_Y else f @ D.T


print("=== 1. axis convention ===")
print(f"  AXIS_Y={hop.AXIS_Y} (rows, even legs)   AXIS_X={hop.AXIS_X} (cols, odd legs)")
probe = np.zeros((n, n))
probe[:, 4] = 1.0  # a single column -> varies along x only
hp = ho.field_to_hybrid(torch.from_numpy(probe), L, N, bd_first=2**N, bd_remaining=2**L, eps=1e-15)
dx_op = hop.derivative_mpo(L, hop.AXIS_X, hop.D1_CENTRAL_2)
dy_op = hop.derivative_mpo(L, hop.AXIS_Y, hop.D1_CENTRAL_2)
gx = ho.hybrid_to_field(ho.hybrid_apply(dx_op, hp, N), L).numpy()
gy = ho.hybrid_to_field(ho.hybrid_apply(dy_op, hp, N), L).numpy()
print(f"  d/dx of a single column: |out|={np.abs(gx).max():.3f} (expect nonzero)")
print(f"  d/dy of a single column: |out|={np.abs(gy).max():.3e} (expect zero)")
check("AXIS_X differentiates across columns", 0.0 if np.abs(gx).max() > 0.1 else 1.0, 0.0)
check("AXIS_Y leaves a column-constant field alone", np.abs(gy).max(), 1e-14)

print("\n=== 2. derivative / laplacian MPOs vs dense ===")
cases = [
    ("d/dx  central 2nd", hop.D1_CENTRAL_2),
    ("d/dx  central 4th", hop.D1_CENTRAL_4),
    ("d2/dx2 central 2nd", hop.D2_CENTRAL_2),
    ("d2/dx2 central 4th", hop.D2_CENTRAL_4),
    ("d/dx  upwind (+)", hop.D1_UPWIND_POS),
    ("d/dx  upwind (-)", hop.D1_UPWIND_NEG),
]
for label, coef in cases:
    for axis, nm in [(hop.AXIS_X, "x"), (hop.AXIS_Y, "y")]:
        op = hop.derivative_mpo(L, axis, coef)
        got = ho.hybrid_to_field(ho.hybrid_apply(op, hyb, N), L).numpy()
        check(f"{label} along {nm}", rel(got, apply_dense(coef, axis)), 1e-11)

for label, coef in [("2nd order", hop.D2_CENTRAL_2), ("4th order", hop.D2_CENTRAL_4)]:
    lap = hop.laplacian_mpo(L, coef)
    got = ho.hybrid_to_field(ho.hybrid_apply(lap, hyb, N), L).numpy()
    ref = apply_dense(coef, hop.AXIS_X) + apply_dense(coef, hop.AXIS_Y)
    check(f"laplacian, {label}", rel(got, ref), 1e-11)

print("\n=== 3. dx scaling ===")
op = hop.derivative_mpo(L, hop.AXIS_X, hop.D1_CENTRAL_2, dx=0.25)
got = ho.hybrid_to_field(ho.hybrid_apply(op, hyb, N), L).numpy()
check("derivative_mpo divides by dx", rel(got, apply_dense(hop.D1_CENTRAL_2, hop.AXIS_X) / 0.25), 1e-11)
lap = hop.laplacian_mpo(L, dx=0.25)
got = ho.hybrid_to_field(ho.hybrid_apply(lap, hyb, N), L).numpy()
ref = (apply_dense(hop.D2_CENTRAL_2, hop.AXIS_X) + apply_dense(hop.D2_CENTRAL_2, hop.AXIS_Y)) / 0.25**2
check("laplacian_mpo divides by dx^2", rel(got, ref), 1e-11)

print("\n=== 4. MPO algebra ===")
A = hop.derivative_mpo(L, hop.AXIS_X, hop.D1_CENTRAL_2)
B = hop.derivative_mpo(L, hop.AXIS_Y, hop.D1_CENTRAL_2)
Ad, Bd = hop.mpo_to_dense(A).numpy(), hop.mpo_to_dense(B).numpy()
check("mpo_add", rel(hop.mpo_to_dense(hop.mpo_add(A, B)).numpy(), Ad + Bd), 1e-12)
check("mpo_scale", rel(hop.mpo_to_dense(hop.mpo_scale(A, -3.0)).numpy(), -3.0 * Ad), 1e-12)
check("mpo_compose", rel(hop.mpo_to_dense(hop.mpo_compose(A, B)).numpy(), Ad @ Bd), 1e-11)
C = hop.mpo_compose(A, B)
check("mpo_compress lossless", rel(hop.mpo_to_dense(hop.mpo_compress(C, eps=1e-14)).numpy(),
                                   hop.mpo_to_dense(C).numpy()), 1e-11)
print(f"  compose bond {C.max_rank} -> compressed {hop.mpo_compress(C, eps=1e-14).max_rank}")

print("\n  -- mpo_compose matches hybrid_apply twice --")
seq = ho.hybrid_apply(A, ho.hybrid_apply(B, hyb, N), N)
comp = ho.hybrid_apply(hop.mpo_compress(hop.mpo_compose(A, B), eps=1e-14), hyb, N)
check("compose(A,B) f == A(B f)", rel(ho.contract(comp).numpy(), ho.contract(seq).numpy()), 1e-11)

print("\n  -- construct_toeplitz always returns bond 9; compression recovers the true rank --")
for label, coef in [("2-term (d/dx)", hop.D1_CENTRAL_2), ("3-term (d2/dx2)", hop.D2_CENTRAL_2),
                    ("5-term (4th order)", hop.D2_CENTRAL_4)]:
    raw = hop.promote_2d(hop.toeplitz_1d(L, coef, "periodic"), hop.AXIS_X, L)
    print(f"     {label:20s} raw bond {raw.max_rank:2d} -> {hop.mpo_compress(raw, eps=1e-14).max_rank}")

print("\n=== 5. seam projector ===")
for Nh in [2, 4, 6]:
    for axis, nm in [(hop.AXIS_X, "x/cols"), (hop.AXIS_Y, "y/rows")]:
        P = hop.seam_projector(L, Nh, axis)
        got = ho.hybrid_to_field(
            ho.hybrid_apply(P, ho.field_to_hybrid(F, L, Nh, bd_first=2**Nh,
                                                  bd_remaining=2**L, eps=1e-15), Nh), L).numpy()
        mask = np.zeros((n, n))
        idx = ho.seam_indices(L, Nh).numpy()
        if axis == hop.AXIS_X:
            mask[:, idx] = 1.0
        else:
            mask[idx, :] = 1.0
        check(f"seam_projector N={Nh} {nm}", rel(got, f * mask), 1e-12)
print(f"  bond dimension of seam_projector: {hop.seam_projector(L, 4, hop.AXIS_X).max_rank} (product operator)")

print("\n=== 6. isotropic seam smoother vs the generate_data.py reference ===")
for Nh in [2, 4, 6]:
    S = hop.seam_smoother_mpo(L, Nh)
    hs = ho.field_to_hybrid(F, L, Nh, bd_first=2**Nh, bd_remaining=2**L, eps=1e-15)
    got = ho.hybrid_to_field(ho.hybrid_apply(S, hs, Nh), L).numpy()
    ref = hop.seam_smooth_dense(F, L, Nh).numpy()
    check(f"seam_smoother_mpo N={Nh} == dense _int64_xy", rel(got, ref), 1e-11)
    print(f"     N={Nh}: MPO bond dim {S.max_rank}, seam period {ho.seam_period(L, Nh)}")

print("\n  -- isotropy: x and y are treated identically --")
S = hop.seam_smoother_mpo(L, 4)
hs = ho.field_to_hybrid(F, L, 4, bd_first=2**4, bd_remaining=2**L, eps=1e-15)
got = ho.hybrid_to_field(ho.hybrid_apply(S, hs, 4), L).numpy()
hsT = ho.field_to_hybrid(F.T.contiguous(), L, 4, bd_first=2**4, bd_remaining=2**L, eps=1e-15)
gotT = ho.hybrid_to_field(ho.hybrid_apply(S, hsT, 4), L).numpy()
check("S(F^T) == S(F)^T", rel(gotT, got.T), 1e-11)

print("\n  -- the smoother only touches seam lines --")
idx = ho.seam_indices(L, 4).numpy()
touched = np.abs(got - f) > 1e-12
off = touched.copy()
off[idx, :] = False
off[:, idx] = False
check("interior cells untouched", float(off.sum()), 0.0)
frac = touched.sum() / touched.size
print(f"     cells modified: {touched.sum()}/{touched.size} ({100 * frac:.1f}%)")

print("\n  -- not idempotent (a correction, not a projection) --")
once = ho.hybrid_apply(S, hs, 4)
twice = ho.hybrid_apply(S, once, 4)
d = rel(ho.contract(twice).numpy(), ho.contract(once).numpy())
print(f"     ||S^2 f - S f|| / ||S f|| = {d:.2e}  -> apply exactly once per step")

print("\n=== 7. L=6, N=6 ===")
L2, N2 = 6, 6
n2 = 2**L2
x2, y2 = np.meshgrid(np.arange(n2), np.arange(n2), indexing="xy")
f2 = np.sin(2 * np.pi * x2 / n2) * np.cos(4 * np.pi * y2 / n2) + 0.1 * np.cos(10 * np.pi * x2 / n2)
F2 = torch.from_numpy(f2)
h2 = ho.field_to_hybrid(F2, L2, N2, bd_first=2**N2, bd_remaining=2**L2, eps=1e-15)
lap2 = hop.laplacian_mpo(L2)
got = ho.hybrid_to_field(ho.hybrid_apply(lap2, h2, N2), L2).numpy()


def dense_1d_L(coef, m):
    M = np.zeros((m, m))
    for i in range(m):
        for k, v in coef.items():
            if v:
                M[i, (i + k) % m] += v
    return M


D2 = dense_1d_L(hop.D2_CENTRAL_2, n2)
check("L=6 N=6 laplacian", rel(got, D2 @ f2 + f2 @ D2.T), 1e-11)
S2 = hop.seam_smoother_mpo(L2, N2)
got = ho.hybrid_to_field(ho.hybrid_apply(S2, h2, N2), L2).numpy()
check("L=6 N=6 seam smoother", rel(got, hop.seam_smooth_dense(F2, L2, N2).numpy()), 1e-11)
print(f"  seam smoother bond dim at L=6 N=6: {S2.max_rank}")

print("\n" + "=" * 66)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    sys.exit(1)
print("all checks passed")
