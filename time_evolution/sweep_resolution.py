"""How does the hybrid's advantage over a standard TT scale with resolution?

The practical question at production scale is not "what error do I get at a
fixed budget" but "what budget do I need for a fixed error". This sweeps
resolution and reports, for each encoding, the compression ratio rho required to
reach a target accuracy -- and how much of that budget the dense head saves.

Resolution is varied at fixed ell_B/dx (kappa ~ dx^2), so every L resolves the
same range of scales and a finer grid genuinely adds structure rather than
merely over-resolving what is already there. At fixed kappa the trend looks
considerably stronger, which is why it is not the number quoted.

Representation only. The companion evolution runs (plot_hybrid_vs_standard.py,
plot_scaleup.py) establish that evolution error tracks representation error
within a factor of ~2-5 for every encoding at L=8 and L=9, so the *scaling* in L
is the same either way; measuring it by representation costs minutes rather than
weeks. Evolving at L=11 to a matched rho would need bond dimensions ~2^L and
cost ~8^L, which is the whole reason the scaling matters.

Dense fields are cached in results/ since generating them dominates the runtime.

Writes figures/resolution_scaling.png.
"""
import argparse
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hybrid_ops as ho
import hybrid_evolve as hev
from run_decaying_cellular import grid, initial_field, build_velocity

RESULTS = os.path.join(_HERE, "results")
FIGS = os.path.join(_HERE, "figures")

T_END = 8.0
KAPPA_REF, L_REF = 5.0e-3, 8          # kappa ~ dx^2 anchored here
TARGET_ERRS = (1e-1, 1e-2, 1e-3)
ERR_COLOR = {1e-1: "#1f77b4", 1e-2: "#ff7f0e", 1e-3: "#2ca02c"}


def kappa_for(L):
    """kappa ~ dx^2, so ell_B/dx is the same at every resolution."""
    return KAPPA_REF * 4.0 ** (L_REF - L)


def cached_field(L):
    path = os.path.join(RESULTS, f"field_L{L}_T{T_END:g}.npy")
    if os.path.exists(path):
        return np.load(path)
    n, dx, _, _ = grid(L)
    kappa = kappa_for(L)
    ux, uy = build_velocity(L, 6, dx)
    uxd = ho.hybrid_to_field(ux, L).numpy()
    uyd = ho.hybrid_to_field(uy, L).numpy()
    dt = hev.stable_dt(1.0, 1.0, kappa, dx, cfl=0.2, diff_number=0.1)
    f = initial_field(L)
    rhs = hev.dense_variable_rhs(uxd, uyd, dx, kappa=kappa, form="skew")
    for _ in range(int(round(T_END / dt))):
        f = hev.dense_step(f, rhs, dt, "ssprk3")
    os.makedirs(RESULTS, exist_ok=True)
    np.save(path, f)
    return f


def bd_ladder(L):
    """Bond caps spanning the useful range; chi must grow ~2^L for fixed rho."""
    base = [2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
    if L >= 9:
        base += [96, 128]
    if L >= 10:
        base += [192]
    return base


def curve(f, L, N):
    """(rho, L2 error) pairs, sorted by rho."""
    tf = torch.from_numpy(np.ascontiguousarray(f))
    nrm = np.linalg.norm(f)
    pts = []
    for bd in bd_ladder(L):
        h = ho.field_to_hybrid(tf, L, N, bd_first=bd, bd_remaining=bd)
        g = ho.hybrid_to_field(h, L).numpy()
        pts.append((ho.num_params(h) / 4 ** L, np.linalg.norm(g - f) / nrm))
    c = np.array(pts)
    return c[np.argsort(c[:, 0])]


def rho_for_error(c, target):
    """Smallest rho reaching `target`, by log-log interpolation of err(rho)."""
    rho, err = c[:, 0], c[:, 1]
    # err decreases with rho; interpolate rho as a function of (decreasing) err
    o = np.argsort(err)
    if target < err[o][0] or target > err[o][-1]:
        return np.nan
    return float(np.exp(np.interp(np.log(target), np.log(err[o]),
                                  np.log(rho[o]))))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--L", type=int, nargs="+", default=[7, 8, 9, 10])
    ap.add_argument("--N", type=int, nargs="+", default=None)
    a = ap.parse_args()
    os.makedirs(FIGS, exist_ok=True)

    table = {}        # (L, N) -> curve
    for L in a.L:
        f = cached_field(L)
        Ns = a.N if a.N else [N for N in range(0, min(2 * L, 15), 2)]
        for N in Ns:
            table[(L, N)] = curve(f, L, N)
        print(f"L={L} done ({2**L}^2, kappa={kappa_for(L):.3e})", flush=True)

    Ls = sorted(a.L)
    # best hybrid at each (L, target): the head size is a free design choice
    best = {}
    for L in Ls:
        Ns = sorted({N for (LL, N) in table if LL == L})
        for tgt in TARGET_ERRS:
            rhos = {N: rho_for_error(table[(L, N)], tgt) for N in Ns}
            std = rhos.get(0, np.nan)
            hyb = {N: r for N, r in rhos.items() if N > 0 and np.isfinite(r)}
            if hyb:
                bN = min(hyb, key=hyb.get)
                best[(L, tgt)] = (std, hyb[bN], bN)
            else:
                best[(L, tgt)] = (std, np.nan, None)

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))

    ax = axes[0]
    for tgt in TARGET_ERRS:
        s = [best[(L, tgt)][0] for L in Ls]
        h = [best[(L, tgt)][1] for L in Ls]
        ax.semilogy(Ls, s, "--o", ms=4, lw=1.3, color=ERR_COLOR[tgt],
                    label=rf"standard, err$={tgt:g}$")
        ax.semilogy(Ls, h, "-s", ms=4, lw=1.6, color=ERR_COLOR[tgt],
                    label=rf"best hybrid, err$={tgt:g}$")
    ax.set_xlabel("$L$"); ax.set_ylabel(r"$\rho$ needed")
    ax.set_xticks(Ls)
    ax.legend(fontsize=6); ax.set_title("budget for a fixed accuracy", fontsize=9)

    ax = axes[1]
    for tgt in TARGET_ERRS:
        g = [best[(L, tgt)][0] / best[(L, tgt)][1] for L in Ls]
        ax.semilogy(Ls, g, "-o", ms=4, lw=1.6, color=ERR_COLOR[tgt],
                    label=rf"err$={tgt:g}$")
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.set_xlabel("$L$"); ax.set_ylabel(r"$\rho_{\rm std}/\rho_{\rm hybrid}$")
    ax.set_xticks(Ls)
    ax.legend(fontsize=7); ax.set_title("parameters saved by the head", fontsize=9)

    ax = axes[2]
    for tgt in TARGET_ERRS:
        bN = [best[(L, tgt)][2] for L in Ls]
        ax.plot(Ls, bN, "-o", ms=4, lw=1.6, color=ERR_COLOR[tgt],
                label=rf"err$={tgt:g}$")
    ax.plot(Ls, Ls, ":", color="0.5", lw=1.2, label="$N=L$")
    ax.set_xlabel("$L$"); ax.set_ylabel("best $N$")
    ax.set_xticks(Ls)
    ax.legend(fontsize=7); ax.set_title("best head size grows slowly", fontsize=9)

    fig.suptitle(r"Hybrid vs standard TT across resolution, at fixed "
                 r"$\ell_B/\Delta x$ (representation)", fontsize=10, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGS, "resolution_scaling.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(out)

    print(f"\n{'L':>3} {'grid':>7} {'target':>8} {'rho std':>9} {'rho hyb':>9} "
          f"{'best N':>7} {'saving':>8}")
    for L in Ls:
        for tgt in TARGET_ERRS:
            s, h, bN = best[(L, tgt)]
            sv = s / h if np.isfinite(s) and np.isfinite(h) else np.nan
            print(f"{L:>3} {2**L:>5}^2 {tgt:>8.0e} {s:>9.4f} {h:>9.4f} "
                  f"{str(bN):>7} {sv:>7.1f}x")
    np.savez_compressed(os.path.join(RESULTS, "resolution_scaling.npz"),
                        **{f"L{L}_N{N}": c for (L, N), c in table.items()})


if __name__ == "__main__":
    main()
