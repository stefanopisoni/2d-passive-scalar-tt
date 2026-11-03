"""Steady state of the forced cellular problem, and the approach to it.

With a steady velocity and a steady source the long-time state is a fixed
point: it solves the linear system  A theta = -f,  where A is exactly the dense
solver's operator (skew transport, second-order central stencils, periodic).
Two modes:

  solve  assemble A as a sparse matrix and solve directly. A annihilates the
         constants (periodic, discretely divergence-free u) and f has zero
         mean, so one value is pinned and the mean removed afterwards; the mean
         is conserved by the dynamics and starts at zero.
  probe  step the dense solver from the usual initial condition and record
         variance, dissipation and production every 0.5 time units, to measure
         how long the approach to that fixed point takes.

Measured at 256^2 (kappa=5e-3): the probe settles to <theta^2> = 0.4804, the
solve gives 0.480383, the assembled A matches the time-stepper's operator to
3e-16, and the slowest mode decays on a time ~53 -- so T=60 ends 6% above the
steady value.

    python forced_steady.py solve --L 10 --kappa 3.125e-4
    python forced_steady.py probe --L 9 --kappa 1.25e-3 --T 800
"""
import argparse
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
torch.set_num_threads(int(os.environ.get("PSCALAR_THREADS", "1")))

import hybrid_ops as ho
import hybrid_operators as hop
import hybrid_evolve as hev
import run_decaying_cellular as R

RESULTS = os.environ.get("PSCALAR_RESULTS", os.path.join(_HERE, "results"))


def velocity(L):
    """The discrete cellular velocity, as both solvers see it."""
    n, dx, _, _ = R.grid(L)
    ux, uy = R.build_velocity(L, 10 if 2 * L > 10 else 0, dx)
    return ho.hybrid_to_field(ux, L).numpy(), ho.hybrid_to_field(uy, L).numpy(), dx


def shift(n, k, axis):
    """(S f)[i] = f[i + k] along ``axis``, periodic, row-major (y, x) --
    the sparse form of the np.roll in _apply_stencil_dense."""
    idx = np.arange(n * n).reshape(n, n)
    src = np.roll(idx, -k, axis=axis).ravel()
    return sp.csr_matrix((np.ones(n * n), (idx.ravel(), src)), shape=(n * n,) * 2)


def operator(L, kappa, ux, uy, dx):
    n = 2 ** L
    Dx = 0.5 * (shift(n, 1, hop.AXIS_X) - shift(n, -1, hop.AXIS_X)) / dx
    Dy = 0.5 * (shift(n, 1, hop.AXIS_Y) - shift(n, -1, hop.AXIS_Y)) / dx
    D2 = (shift(n, 1, 0) + shift(n, -1, 0) + shift(n, 1, 1) + shift(n, -1, 1)
          - 4 * sp.identity(n * n)) / dx ** 2
    Ux, Uy = sp.diags(ux.ravel()), sp.diags(uy.ravel())
    adv = -(Ux @ Dx + Uy @ Dy)
    con = -(Dx @ Ux + Dy @ Uy)
    return (0.5 * (adv + con) + kappa * D2).tocsr()


def solve(L, kappa, amp, kf):
    n = 2 ** L
    ux, uy, dx = velocity(L)
    f = R.forcing_field(L, amp, kf)
    rhs = hev.dense_variable_rhs(ux, uy, dx, kappa=kappa, form="skew", forcing=f)
    A = operator(L, kappa, ux, uy, dx)
    g = np.random.default_rng(0).standard_normal((n, n))
    ref = rhs(g) - f
    op_check = np.abs((A @ g.ravel()).reshape(n, n) - ref).max() / np.abs(ref).max()
    b = -f.ravel().copy()
    A = A.tolil()
    A[0, :] = 0
    A[0, 0] = 1.0
    b[0] = 0.0
    t0 = time.time()
    th = spl.spsolve(A.tocsc(), b).reshape(n, n)
    th -= th.mean()
    residual = np.abs(rhs(th)).max() / np.abs(f).max()
    out = os.path.join(RESULTS, f"steady_L{L}.npy")
    np.save(out, th)
    print(f"L={L} kappa={kappa}: var={th.var():.6f} op_check={op_check:.1e} "
          f"residual={residual:.1e} solve={time.time()-t0:.1f}s -> {out}")


def probe(L, kappa, amp, kf, T):
    ux, uy, dx = velocity(L)
    f = R.forcing_field(L, amp, kf)
    rhs = hev.dense_variable_rhs(ux, uy, dx, kappa=kappa, form="skew", forcing=f)
    umax = float(max(np.abs(ux).max(), np.abs(uy).max()))
    dt = hev.stable_dt(umax, umax, kappa, dx)
    n_steps = int(round(T / dt))
    every = max(1, int(round(0.5 / dt)))
    th = R.initial_field(L)
    rows = []
    t0 = time.time()
    for i in range(n_steps + 1):
        if i % every == 0 or i == n_steps:
            gx = hev._apply_stencil_dense(th, hop.D1_CENTRAL_2, hop.AXIS_X) / dx
            gy = hev._apply_stencil_dense(th, hop.D1_CENTRAL_2, hop.AXIS_Y) / dx
            rows.append((i * dt, th.var(), kappa * (gx**2 + gy**2).mean(),
                         (th * f).mean()))
        if i < n_steps:
            th = hev.dense_step(th, rhs, dt, "ssprk3")
    out = os.path.join(RESULTS, f"probe_forced_L{L}_T{T:g}.npy")
    np.save(out, np.array(rows))
    print(f"L={L} kappa={kappa} T={T} steps={n_steps} "
          f"wall={(time.time()-t0)/3600:.2f} h -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=("solve", "probe"))
    p.add_argument("--L", type=int, required=True)
    p.add_argument("--kappa", type=float, required=True)
    p.add_argument("--T", type=float, default=800.0)
    p.add_argument("--forcing-amp", type=float, default=0.8)
    p.add_argument("--forcing-k", type=int, default=4)
    a = p.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    if a.mode == "solve":
        solve(a.L, a.kappa, a.forcing_amp, a.forcing_k)
    else:
        probe(a.L, a.kappa, a.forcing_amp, a.forcing_k, a.T)
