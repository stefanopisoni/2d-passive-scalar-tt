"""How big should the dense head be, and how does that scale with resolution?

The head costs 2^N * chi_0 parameters outright, whatever the grid size, and buys
an exact representation of the 2^(N/2) x 2^(N/2) coarse blocks. Too small and it
buys nothing; too large and it consumes the whole budget, leaving no rank for
the tail. The optimum is therefore interior, and where it sits decides how the
hybrid encoding should be configured at production resolution.

Resolution is varied two ways. At fixed kappa a finer grid simply over-resolves
the same physical structure (the Batchelor scale is 1.4 cells at L=7 but 5.8 at
L=9), which flatters the finer grids. The honest scaling fixes ell_B/dx by
taking kappa ~ dx^2, so every L resolves the same number of scales; both are
computed here and the second is the one to quote.

This sweeps N at three resolutions on the evolved cellular field. It is
representation only -- no time stepping -- which is legitimate here because the
companion experiment (plot_hybrid_vs_standard.py) establishes that the
evolution error tracks the representation error within a factor of 2-5 for every
encoding, so the *ordering* in N is the same either way. Representation-only
costs seconds instead of hours.

Writes figures/head_size_sweep.png.
"""
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

FIGS = os.path.join(_HERE, "figures")
RHO_TARGETS = (0.05, 0.10, 0.20)
BDS = (2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64)
L_VALUES = (7, 8, 9)
T_END = 8.0
KAPPA = 5e-3
# kappa ~ dx^2 holds ell_B/dx fixed, so each L resolves the same range of scales
KAPPA_FIXED_BATCHELOR = {7: 2.0e-2, 8: 5.0e-3, 9: 1.25e-3}
SCALINGS = ((r"fixed $\kappa$", None),
            (r"fixed $\ell_B/\Delta x$", KAPPA_FIXED_BATCHELOR))


def evolved_field(L, T=T_END, kappa=KAPPA):
    """The dense solution at time T -- the field the encodings have to carry."""
    n, dx, _, _ = grid(L)
    ux, uy = build_velocity(L, 6, dx)
    uxd = ho.hybrid_to_field(ux, L).numpy()
    uyd = ho.hybrid_to_field(uy, L).numpy()
    dt = hev.stable_dt(1.0, 1.0, kappa, dx, cfl=0.2, diff_number=0.1)
    f = initial_field(L)
    rhs = hev.dense_variable_rhs(uxd, uyd, dx, kappa=kappa, form="skew")
    for _ in range(int(round(T / dt))):
        f = hev.dense_step(f, rhs, dt, "ssprk3")
    return f


def error_at_rho(f, L, N, targets=RHO_TARGETS, bds=BDS):
    """Representation error interpolated onto fixed compression ratios."""
    tf = torch.from_numpy(np.ascontiguousarray(f))
    nrm = np.linalg.norm(f)
    pts = []
    for bd in bds:
        h = ho.field_to_hybrid(tf, L, N, bd_first=bd, bd_remaining=bd)
        g = ho.hybrid_to_field(h, L).numpy()
        pts.append((ho.num_params(h) / 4 ** L, np.linalg.norm(g - f) / nrm))
    c = np.array(pts)
    o = np.argsort(c[:, 0])
    return np.exp(np.interp(np.log(targets), np.log(c[o, 0]), np.log(c[o, 1]),
                            left=np.nan, right=np.nan))


def main():
    os.makedirs(FIGS, exist_ok=True)
    fig, axes = plt.subplots(len(SCALINGS), len(RHO_TARGETS),
                             figsize=(10.5, 6.2), squeeze=False)
    cm = plt.cm.viridis(np.linspace(0, 0.75, len(L_VALUES)))
    summary = {}
    for i, (sname, kmap) in enumerate(SCALINGS):
        for c, L in zip(cm, L_VALUES):
            kappa = KAPPA if kmap is None else kmap[L]
            f = evolved_field(L, kappa=kappa)
            Ns = [N for N in range(0, min(2 * L, 15), 2)]
            err = np.array([error_at_rho(f, L, N) for N in Ns])
            summary[(i, L)] = (Ns, err, kappa)
            for j in range(len(RHO_TARGETS)):
                axes[i][j].semilogy(Ns, err[:, j], "-o", ms=4, lw=1.4, color=c,
                                    label=rf"$L={L}$ (${2**L}^2$)")
        for j, rho in enumerate(RHO_TARGETS):
            axes[i][j].set_xlabel("$N$ (head legs)")
            axes[i][j].set_title(rf"{sname},  $\rho={rho:g}$", fontsize=9)
            axes[i][j].legend(fontsize=7)
        axes[i][0].set_ylabel(r"$L_2$ representation error")
    fig.suptitle("Best head size, and how its advantage scales with resolution",
                 fontsize=10, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = os.path.join(FIGS, "head_size_sweep.png")
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(out)

    for i, (sname, _) in enumerate(SCALINGS):
        print(f"\n--- {sname.replace('$','').replace(chr(92),'')} ---")
        print(f"{'L':>3} {'kappa':>9} {'rho':>6} {'best N':>7} {'blocks':>8} "
              f"{'err(N=0)':>10} {'err(best)':>10} {'gain':>7}")
        for L in L_VALUES:
            Ns, err, kappa = summary[(i, L)]
            for j, rho in enumerate(RHO_TARGETS):
                col = err[:, j]
                if np.all(np.isnan(col)):
                    continue
                b = int(np.nanargmin(col))
                print(f"{L:>3} {kappa:>9.2e} {rho:>6.2f} {Ns[b]:>7} "
                      f"{2**(Ns[b]//2):>6}^2 {col[0]:>10.2e} {col[b]:>10.2e} "
                      f"{col[0]/col[b]:>6.1f}x")


if __name__ == "__main__":
    main()
