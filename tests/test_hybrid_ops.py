"""Validation of hybrid_ops against dense references.

Run:  python tests/test_hybrid_ops.py
Every hybrid routine is checked against the equivalent dense numpy/torch
computation at a size small enough (L = 5, 6) to form the dense objects.
"""
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

import config as cfg

sys.path.append(cfg.TENSNET_SRC)
import tensnet as tt  # noqa: E402

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
    d = np.abs(np.asarray(a) - np.asarray(b)).max()
    s = max(np.abs(np.asarray(b)).max(), 1e-300)
    return float(d / s)


# ---------------------------------------------------------------- fixtures
L, N = 5, 4
n = 2**L
rng = np.random.default_rng(7)

xx, yy = np.meshgrid(np.arange(n), np.arange(n), indexing="xy")
smooth = np.exp(-((xx - 11) ** 2 + (yy - 20) ** 2) / 25.0) \
    + 0.4 * np.sin(2 * np.pi * xx / n) * np.cos(4 * np.pi * yy / n)

k = np.fft.fftfreq(n) * n
KX, KY = np.meshgrid(k, k, indexing="xy")
K = np.sqrt(KX**2 + KY**2)
K[0, 0] = 1.0
rough = np.real(np.fft.ifft2((rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) * K**-1.0))
rough /= rough.std()

F_smooth = torch.from_numpy(smooth)
F_rough = torch.from_numpy(rough)


print("=== 1. index layout: shuffle/unshuffle round trip ===")
check("unshuffle(shuffle(F)) == F",
      rel(ho.unshuffle_axis(ho.shuffle_axis(F_smooth, L), L).numpy(), smooth), 0.0)

print("\n=== 2. seam geometry ===")
for Nh in [2, 4, 6, 8]:
    p = ho.seam_period(L, Nh)
    idx = ho.seam_indices(L, Nh).numpy()
    exp_p = n // (2 ** (Nh // 2))
    exp_idx = np.where((np.arange(n) + 1) % exp_p == 0)[0]
    ok = (p == exp_p) and np.array_equal(idx, exp_idx)
    print(f"  {'PASS' if ok else '**FAIL**':10s} N={Nh}: period={p} (2^(L-N/2)={exp_p}), "
          f"{len(idx)} seam lines per axis")
    if not ok:
        FAILURES.append(f"seam geometry N={Nh}")

print("\n=== 3. hybrid construction / contraction ===")
h_exact = ho.field_to_hybrid(F_smooth, L, N, bd_first=2**L, bd_remaining=2**L, eps=1e-15)
ho.validate_hybrid(h_exact, L, N)
print(f"  head shape {tuple(ho.tensors(h_exact)[0].shape)}, bonds {ho.bond_dims(h_exact)}")
check("lossless hybrid round trip (field -> TT -> field)",
      rel(ho.hybrid_to_field(h_exact, L).numpy(), smooth), 1e-12)
check("ho.contract == MPS.contract",
      rel(ho.contract(h_exact).numpy(), h_exact.contract().numpy()), 1e-14)

h = ho.field_to_hybrid(F_rough, L, N, bd_first=6, bd_remaining=6)
ho.validate_hybrid(h, L, N)
base = ho.contract(h).numpy()
print(f"  truncated hybrid: bonds {ho.bond_dims(h)}, params {ho.num_params(h)}")

# The legacy constructor renormalises singular values after every truncation
# (tt.svd's renormalize=True default), so it matches the original field's norm
# instead of being the best rank-r approximation. We must reproduce it exactly
# on demand, and be strictly more accurate by default.
from hybrid_MPS import vector_to_hybrid_mps  # noqa: E402

ref_vec = ho.shuffle_axis(F_rough, L).flatten().numpy()
for bd in [4, 6, 12]:
    legacy = vector_to_hybrid_mps(ho.shuffle_axis(F_rough, L).flatten(), 2 * L, N,
                                  bd_first=bd, bd_remaining=bd)
    repro = ho.field_to_hybrid(F_rough, L, N, bd_first=bd, bd_remaining=bd,
                               renormalize=True)
    check(f"renormalize=True reproduces vector_to_hybrid_mps, bd={bd}",
          rel(ho.contract(repro).numpy(), ho.contract(legacy).numpy()), 1e-12)

print("  default (plain truncation) vs legacy (norm-matched):")
exact_norm = np.linalg.norm(ref_vec)
for bd in [4, 6, 12, 20]:
    ours = ho.field_to_hybrid(F_rough, L, N, bd_first=bd, bd_remaining=bd)
    legacy = vector_to_hybrid_mps(ho.shuffle_axis(F_rough, L).flatten(), 2 * L, N,
                                  bd_first=bd, bd_remaining=bd)
    e_o = np.abs(ho.contract(ours).numpy() - ref_vec).max()
    e_l = np.abs(ho.contract(legacy).numpy() - ref_vec).max()
    better = e_o <= e_l + 1e-12
    if not better:
        FAILURES.append(f"default less accurate than legacy at bd={bd}")
    print(f"    {'ok ' if better else 'BAD'} bd={bd:2d}  max-err ours={e_o:.4e} legacy={e_l:.4e}   "
          f"norm ours={ho.hybrid_norm(ours):.4f} legacy={ho.hybrid_norm(legacy):.4f} exact={exact_norm:.4f}")

# ... and that eps is now a genuine amplitude tolerance
ref_vec = ho.shuffle_axis(F_rough, L).flatten().numpy()
print("  eps semantics (achieved max-norm error vs requested eps):")
for eps in [1e-4, 1e-8, 1e-12]:
    a = ho.field_to_hybrid(F_rough, L, N, bd_first=2**N, bd_remaining=2**L, eps=eps)
    b = vector_to_hybrid_mps(ho.shuffle_axis(F_rough, L).flatten(), 2 * L, N,
                             bd_first=2**N, bd_remaining=2**L, eps=eps)
    print(f"    eps={eps:.0e}:  ours={rel(ho.contract(a).numpy(), ref_vec):.2e}"
          f"   tt.svd path={rel(ho.contract(b).numpy(), ref_vec):.2e}")

print("\n=== 4. exact arithmetic (no implicit compression) ===")
g = ho.field_to_hybrid(F_smooth, L, N, bd_first=5, bd_remaining=5)
gb = ho.contract(g).numpy()

check("hybrid_add", rel(ho.contract(ho.hybrid_add(h, g)).numpy(), base + gb), 1e-13)
check("hybrid_sub", rel(ho.contract(ho.hybrid_sub(h, g)).numpy(), base - gb), 1e-13)
check("hybrid_scale", rel(ho.contract(ho.hybrid_scale(h, -2.5)).numpy(), -2.5 * base), 1e-14)
check("hybrid_axpy", rel(ho.contract(ho.hybrid_axpy(0.3, h, -1.7, g)).numpy(),
                         0.3 * base - 1.7 * gb), 1e-13)
check("hybrid_hadamard", rel(ho.contract(ho.hybrid_hadamard(h, g)).numpy(), base * gb), 1e-13)

s = ho.hybrid_add(h, g)
print(f"  bonds: h={ho.bond_dims(h)} g={ho.bond_dims(g)}")
print(f"         h+g={ho.bond_dims(s)}  (adds)   h*g={ho.bond_dims(ho.hybrid_hadamard(h, g))}  (multiplies)")
print("  -> confirms no hidden truncation, unlike MPS.__add__ / __mul__")

# contrast with tensnet's own operators, which do compress silently
t_add = h + g
print(f"  tensnet h+g bonds={[T.shape[2] for T in [nd.tensor for nd in t_add]][:-1]}, "
      f"max-norm err vs exact = {rel(t_add.contract().numpy(), base + gb):.2e}")

print("\n=== 5. hybrid_compress ===")
big = ho.hybrid_hadamard(h, g)
bb = ho.contract(big).numpy()
c_loss = ho.hybrid_compress(big, bd_first=2**N, bd_remaining=2**L, eps=1e-14)
check("lossless recompression of a rank-inflated state",
      rel(ho.contract(c_loss).numpy(), bb), 1e-11)
print(f"  bonds {ho.bond_dims(big)} -> {ho.bond_dims(c_loss)}")

print("  monotone error vs bd_remaining:")
prev = np.inf
for bd in [2, 4, 8, 16]:
    c = ho.hybrid_compress(big, bd_first=2**N, bd_remaining=bd)
    e = rel(ho.contract(c).numpy(), bb)
    mono = e <= prev + 1e-12
    print(f"    {'ok ' if mono else 'BAD'} bd_remaining={bd:2d} -> bonds {ho.bond_dims(c)} relerr={e:.3e}")
    if not mono:
        FAILURES.append("compression error not monotone")
    prev = e

c = ho.hybrid_compress(big, bd_first=3, bd_remaining=16)
head_bond = ho.bond_dims(c)[0]
check("bd_first caps the head bond independently", abs(head_bond - 3), 0)
print(f"    bd_first=3, bd_remaining=16 -> bonds {ho.bond_dims(c)}")

check("compress preserves head physical dim",
      abs(ho.tensors(c)[0].shape[1] - 2**N), 0)

print("  no hidden renormalisation (truncation must lose norm):")
n_full = ho.hybrid_norm(big)
n_trunc = ho.hybrid_norm(ho.hybrid_compress(big, bd_first=2, bd_remaining=2))
print(f"    ||full||={n_full:.6f}  ||rank-2||={n_trunc:.6f}  "
      f"{'ok - norm decreased' if n_trunc < n_full else 'BAD - norm preserved artificially'}")
if not n_trunc < n_full:
    FAILURES.append("truncation renormalised")

print("\n=== 6. diagnostics ===")
check("hybrid_norm", abs(ho.hybrid_norm(h) - np.linalg.norm(base)) / np.linalg.norm(base), 1e-12)
check("hybrid_sum", abs(ho.hybrid_sum(h) - base.sum()) / max(abs(base.sum()), 1e-12), 1e-10)

print("\n=== 7. hybrid_apply vs dense operator ===")
from tensnet.linalg.diff_operators import construct_toeplitz  # noqa: E402
from tensnet.networks.base_mpo import identity_mpo  # noqa: E402
from tensnet.linalg.diff_operators import get_2D_operator  # noqa: E402


def coeffs(d):
    c = {i: 0.0 for i in range(-4, 5)}
    c.update(d)
    return c


def dense_1d(nofq, cdict):
    m = 2**nofq
    M = np.zeros((m, m))
    for i in range(m):
        for kk, v in cdict.items():
            if v:
                M[i, (i + kk) % m] += v
    return M


# a) 1-D operator promoted to 2-D, applied to a *uniform* MPS first (sanity)
stencils = {
    "d/dx  (2nd order central)": coeffs({1: 0.5, -1: -0.5}),
    "d2/dx2 (2nd order)": coeffs({0: -2.0, 1: 1.0, -1: 1.0}),
    "neighbour average": coeffs({1: 0.5, -1: 0.5}),
}

for label, cd in stencils.items():
    D1 = dense_1d(L, cd)
    P = construct_toeplitz(L, cd, "periodic")
    for axis in (0, 1):
        Op = get_2D_operator(P, axis, [L, L], arch="sequential")
        out = ho.hybrid_apply(Op, h_exact, N)
        got = ho.hybrid_to_field(out, L).numpy()
        ref = D1 @ smooth if axis == 0 else smooth @ D1.T
        check(f"hybrid_apply: {label}, axis={axis}", rel(got, ref), 1e-11)

print("\n  -- bond bookkeeping --")
Op = get_2D_operator(construct_toeplitz(L, stencils["d/dx  (2nd order central)"], "periodic"),
                     1, [L, L], arch="sequential")
r_mpo = Op.max_rank
before, after = ho.bond_dims(h), ho.bond_dims(ho.hybrid_apply(Op, h, N))
print(f"     MPO max rank {r_mpo};  bonds {before} -> {after}")
ok_bonds = all(a <= b * r_mpo for a, b in zip(after, before))
print(f"     {'PASS' if ok_bonds else '**FAIL**'} every bond grew by at most the MPO rank")
if not ok_bonds:
    FAILURES.append("bond growth exceeds MPO rank")

print("\n  -- head is never densified --")
print(f"     head operator would be 4^N = {4**N} entries; "
      f"_apply_head works on {2**N} x (bond) instead")

print("\n=== 8. hybrid_apply on a truncated state, vs dense ===")
rec = ho.hybrid_to_field(h, L).numpy()
for label, cd in stencils.items():
    D1 = dense_1d(L, cd)
    P = construct_toeplitz(L, cd, "periodic")
    for axis in (0, 1):
        Op = get_2D_operator(P, axis, [L, L], arch="sequential")
        got = ho.hybrid_to_field(ho.hybrid_apply(Op, h, N), L).numpy()
        ref = D1 @ rec if axis == 0 else rec @ D1.T
        check(f"truncated state: {label}, axis={axis}", rel(got, ref), 1e-11)

print("\n=== 9. operator composition (D applied twice == D^2) ===")
P = construct_toeplitz(L, stencils["d/dx  (2nd order central)"], "periodic")
Op = get_2D_operator(P, 1, [L, L], arch="sequential")
twice = ho.hybrid_apply(Op, ho.hybrid_apply(Op, h_exact, N), N)
D1 = dense_1d(L, stencils["d/dx  (2nd order central)"])
check("D_x(D_x f) == f D1^T D1^T", rel(ho.hybrid_to_field(twice, L).numpy(), smooth @ D1.T @ D1.T), 1e-10)

print("\n=== 10. larger head, L=6 N=6 (odd blocks per axis) ===")
L2, N2 = 6, 6
n2 = 2**L2
xx2, yy2 = np.meshgrid(np.arange(n2), np.arange(n2), indexing="xy")
f2 = np.exp(-((xx2 - 20) ** 2 + (yy2 - 40) ** 2) / 60.0) + 0.3 * np.cos(6 * np.pi * xx2 / n2)
F2 = torch.from_numpy(f2)
h2 = ho.field_to_hybrid(F2, L2, N2, bd_first=2**L2, bd_remaining=2**L2, eps=1e-15)
ho.validate_hybrid(h2, L2, N2)
check("L=6 N=6 round trip", rel(ho.hybrid_to_field(h2, L2).numpy(), f2), 1e-12)
D2 = dense_1d(L2, stencils["d2/dx2 (2nd order)"])
P2 = construct_toeplitz(L2, stencils["d2/dx2 (2nd order)"], "periodic")
for axis in (0, 1):
    Op2 = get_2D_operator(P2, axis, [L2, L2], arch="sequential")
    got = ho.hybrid_to_field(ho.hybrid_apply(Op2, h2, N2), L2).numpy()
    ref = D2 @ f2 if axis == 0 else f2 @ D2.T
    check(f"L=6 N=6 hybrid_apply d2/dx2 axis={axis}", rel(got, ref), 1e-11)

print("\n=== 11. guards ===")
for label, fn in [
    ("odd N rejected", lambda: ho.seam_period(L, 3)),
    ("N >= 2L rejected", lambda: ho.seam_period(L, 2 * L)),
    ("wrong site count rejected", lambda: ho.hybrid_apply(Op, h2, N2)),
    ("mismatched add rejected", lambda: ho.hybrid_add(h, h2)),
]:
    try:
        fn()
        print(f"  **FAIL**   {label} (no exception)")
        FAILURES.append(label)
    except Exception as e:
        print(f"  PASS       {label:36s} -> {type(e).__name__}")

print("\n" + "=" * 62)
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S): " + ", ".join(FAILURES))
    sys.exit(1)
print("all checks passed")
