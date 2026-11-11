"""Forced cellular flow at 512^2 and 1024^2: hybrid evolution held at the steady state.

Each hold run starts from the exact dense steady state (forced_steady.py solve),
so the dense reference does not move and any departure is truncation feeding
back through the dynamics. Reads decay_HOLD*.npz from PSCALAR_RESULTS.

    PSCALAR_RESULTS=<dir with forced_hires results> python plot_forced_hires.py
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("PSCALAR_RESULTS", os.path.join(_HERE, "results"))
OUT = os.path.join(_HERE, "figures", "forced_hires.png")

# exact steady <theta^2> from the sparse solve (forced_steady.py)
VAR_SS = {9: 0.495578, 10: 0.497527}
N_COLOR = {0: "#1f77b4", 8: "#ff7f0e", 10: "#2ca02c", 12: "#d62728"}

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "legend.fontsize": 7})


def load(L):
    runs = {}
    for f in glob.glob(os.path.join(RESULTS, f"decay_HOLD{L}_N*_bd*.npz")):
        d = np.load(f)
        runs[(int(d["N"]), int(d["bd"]))] = d
    return runs


def by_head(runs):
    for N in sorted({k[0] for k in runs}):
        yield N, [runs[k] for k in sorted(k for k in runs if k[0] == N)]


def main():
    hold = {L: load(L) for L in (9, 10)}
    fig, axs = plt.subplots(2, 3, figsize=(10, 6))
    for row, L in enumerate((9, 10)):
        n = 2 ** L
        ax_e, ax_v, ax_t = axs[row]
        for N, ds in by_head(hold[L]):
            c = N_COLOR[N]
            rho = [d["tt_params"][-1] / 4.0 ** L for d in ds]
            ax_e.loglog(rho, [d["err_l2"][-1] for d in ds], "o-", color=c,
                        ms=4, label=f"N={N}")
            ax_e.loglog([d["repr_params"][-1] / 4.0 ** L for d in ds],
                        [d["repr_err_l2"][-1] for d in ds], "s--", color=c,
                        ms=3, lw=0.9, mfc="none")
            ax_v.loglog(rho, [abs(d["tt_variance"][-1] / VAR_SS[L] - 1)
                              for d in ds], "o-", color=c, ms=4, label=f"N={N}")
        for (N, bd), d in sorted(hold[L].items()):
            if bd == 16:
                ax_t.semilogy(d["t"], d["err_l2"], color=N_COLOR[N],
                              label=f"N={N}")
                ax_t.semilogy(d["t"], d["repr_err_l2"], color=N_COLOR[N],
                              ls="--", lw=0.9)
        T = float(next(iter(hold[L].values()))["T"])
        ax_e.set(xlabel=r"$\rho$", ylabel=r"relative $L_2$ error at $t=T$",
                 title=f"{n}$^2$, T={T:g}: evolved (o) vs compressed truth (s)")
        ax_v.set(xlabel=r"$\rho$",
                 ylabel=r"$|\langle\theta^2\rangle_{TT}/\langle\theta^2\rangle_{ss}-1|$",
                 title=f"{n}$^2$: steady variance error at $t=T$")
        ax_t.set(xlabel="t", ylabel=r"relative $L_2$ error",
                 title=rf"{n}$^2$, $\chi=16$: evolved (—) vs compressed (--)")
        for ax in (ax_e, ax_v, ax_t):
            ax.legend(loc="best")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT)
    print("->", OUT)


if __name__ == "__main__":
    main()
