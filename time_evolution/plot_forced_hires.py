"""Forced cellular flow at 512^2 and 1024^2: relaxation, and hybrid evolution held at the steady state.

Each hold run starts from the exact dense steady state (forced_steady.py solve),
so the dense reference does not move and any departure is truncation feeding
back through the dynamics. Reads decay_HOLD*/FRC*/LONG*.npz from PSCALAR_RESULTS
and the steady states and dense probes from PSCALAR_STEADY. Writes
figures/forced_hires.png, figures/forced_relaxation.png and tables_forced_hires.tex.

    PSCALAR_RESULTS=<forced_hires dir> PSCALAR_STEADY=<forced_steady dir> \
        python plot_forced_hires.py
"""
import glob
import os

import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.environ.get("PSCALAR_RESULTS", os.path.join(_HERE, "results"))
STEADY = os.environ.get("PSCALAR_STEADY",
                        os.path.join(os.path.dirname(RESULTS), "forced_steady"))
OUT = os.path.join(_HERE, "figures", "forced_hires.png")
OUT_RELAX = os.path.join(_HERE, "figures", "forced_relaxation.png")
OUT_LONG = os.path.join(_HERE, "figures", "forced_long.png")
OUT_TEX = os.path.join(_HERE, "tables_forced_hires.tex")

KAPPA = {8: 5e-3, 9: 1.25e-3, 10: 3.125e-4}
N_COLOR = {0: "#1f77b4", 8: "#ff7f0e", 10: "#2ca02c", 12: "#d62728"}
L_COLOR = {8: "#1f77b4", 9: "#ff7f0e", 10: "#2ca02c"}
# err(T)/err(T/2) above this marks a run whose error had not levelled off
GROWING = 1.3

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "legend.fontsize": 7})


def steady_var(L):
    """Exact steady <theta^2> from the sparse solve."""
    return float(np.load(os.path.join(STEADY, f"steady_L{L}.npy")).var())


VAR_SS = {L: steady_var(L) for L in (8, 9, 10)}


def load(tag, L):
    runs = {}
    for f in glob.glob(os.path.join(RESULTS, f"decay_{tag}{L}_N*_bd*.npz")):
        d = np.load(f)
        runs[(int(d["N"]), int(d["bd"]))] = d
    return runs


def by_head(runs):
    for N in sorted({k[0] for k in runs}):
        yield N, [runs[k] for k in sorted(k for k in runs if k[0] == N)]


def growth(d):
    t, e = d["t"], d["err_l2"]
    return e[-1] / np.interp(t[-1] / 2, t, e)


def relaxation(L):
    """Probe history, relative gap to the steady variance, and the e-folding
    time of its exponential tail."""
    a = np.load(glob.glob(os.path.join(STEADY, f"probe_forced_L{L}_T*.npy"))[0])
    t, gap = a[:, 0], np.abs(a[:, 1] / VAR_SS[L] - 1)
    m = (gap > 1e-6) & (gap < 5e-2) & (t > 50)
    tau = -1 / np.polyfit(t[m], np.log(gap[m]), 1)[0]
    within = {e: (t[np.where(gap > e)[0][-1] + 1]
                  if gap[-1] <= e else np.nan) for e in (1e-2, 1e-3)}
    return t, gap, tau, within


def fig_hold():
    hold = {L: load("HOLD", L) for L in (9, 10)}
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
    fig.savefig(OUT)
    print("->", OUT)
    return hold


def fig_long():
    """1024^2 hold runs extended to T=600, overlaid on their T=200 counterparts."""
    hold, long = load("HOLD", 10), load("LONG", 10)
    if not long:
        return long
    fig, (ax_e, ax_v) = plt.subplots(1, 2, figsize=(8, 3))
    styles = ["-", "-.", ":", (0, (5, 1))]
    for i, ((N, bd), d) in enumerate(sorted(long.items())):
        c, ls = N_COLOR[N], styles[i % len(styles)]
        lab = rf"N={N}, $\chi$={bd}"
        ax_e.semilogy(d["t"], d["err_l2"], color=c, ls=ls, label=lab)
        ax_e.semilogy(d["t"], d["repr_err_l2"], color=c, ls=ls, lw=0.6, alpha=0.6)
        ax_v.semilogy(d["t"], np.abs(d["tt_variance"] / VAR_SS[10] - 1),
                      color=c, ls=ls, label=lab)
        h = hold.get((N, bd))
        if h is not None:
            ax_e.plot(h["t"][-1], h["err_l2"][-1], "o", color=c, ms=4, mfc="none")
    for ax in (ax_e, ax_v):
        ax.axvline(200, color="0.5", lw=0.6, ls="--")
        ax.legend(loc="best")
    ax_e.set(xlabel="t", ylabel=r"relative $L_2$ error",
             title=r"1024$^2$ to T=600: evolved (thick), compressed (thin)")
    ax_v.set(xlabel="t",
             ylabel=r"$|\langle\theta^2\rangle_{TT}/\langle\theta^2\rangle_{ss}-1|$",
             title="Steady variance error")
    fig.tight_layout()
    fig.savefig(OUT_LONG)
    print("->", OUT_LONG)
    return long


def long_table(long):
    """Extended runs: error at T=200 and T=600, ratio to the representation
    error, growth over the second half, and steady-variance error at T=600."""
    hold = load("HOLD", 10)
    lines = [r"\begin{tabular}{rrrcccrc}", r"\toprule",
             r"$N$ & $\chi$ & $\rho$ & $L_2(t=200)$ & $L_2(t=600)$ & ratio & growth & "
             r"variance error\\", r"\midrule"]
    for (N, bd), d in sorted(long.items()):
        h = hold.get((N, bd))
        e200 = np.interp(200.0, d["t"], d["err_l2"]) if h is None else h["err_l2"][-1]
        e, r = d["err_l2"][-1], d["repr_err_l2"][-1]
        lines.append(rf"{N} & {bd} & {d['tt_params'][-1] / 4.0 ** 10:.4f} & {sci(e200)} & "
                     rf"{sci(e)} & {e / r:.1f} & {growth(d):.2f} & "
                     rf"{sci(abs(d['tt_variance'][-1] / VAR_SS[10] - 1))}\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def fig_relaxation():
    fig, (ax_g, ax_s) = plt.subplots(1, 2, figsize=(8, 3))
    rel = {}
    for L in (8, 9, 10):
        t, gap, tau, within = rel[L] = relaxation(L)
        n = 2 ** L
        ax_g.semilogy(t, gap, color=L_COLOR[L],
                      label=rf"{n}$^2$, $\kappa={KAPPA[L]:g}$, $\tau={tau:.0f}$")
        ax_s.semilogy(t * KAPPA[L], gap, color=L_COLOR[L], label=f"{n}$^2$")
    ax_g.set(xlabel="t", ylim=(1e-7, 2),
             ylabel=r"$|\langle\theta^2\rangle/\langle\theta^2\rangle_{ss}-1|$",
             title="Dense approach to the steady state")
    ax_s.set(xlabel=r"$\kappa t$", ylim=(1e-7, 2),
             title=r"Against $\kappa t$: the tail rate scales as $\kappa$")
    for ax in (ax_g, ax_s):
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT_RELAX)
    print("->", OUT_RELAX)
    return rel


def sci(x):
    if not np.isfinite(x):
        return "--"
    m, e = f"{x:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


def grid_table(runs, chis, mark_growing):
    """Rows N, columns chi; each cell the evolved L2 error at t=T with its ratio
    to the representation error in parentheses."""
    lines = [r"\begin{tabular}{r" + "c" * len(chis) + "}", r"\toprule",
             r"$N$ & " + " & ".join(rf"$\chi={c}$" for c in chis) + r"\\",
             r"\midrule"]
    for N in sorted({k[0] for k in runs}):
        cells = []
        for c in chis:
            d = runs.get((N, c))
            if d is None:
                cells.append("--")
                continue
            e, r = d["err_l2"][-1], d["repr_err_l2"][-1]
            dag = r"$^\dagger$" if mark_growing and growth(d) > GROWING else ""
            cells.append(f"{sci(e)} ({e / r:.1f}){dag}")
        lines.append(f"{N} & " + " & ".join(cells) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def rho_at(ds, target):
    """Budget at which an (N fixed, chi swept) curve reaches the target error,
    by log-log interpolation; None if the curve does not bracket it."""
    rho = np.array([d["tt_params"][-1] / 4.0 ** int(d["L"]) for d in ds])
    err = np.array([d["err_l2"][-1] for d in ds])
    for i in range(len(ds) - 1):
        if err[i] >= target >= err[i + 1]:
            w = np.log(target / err[i]) / np.log(err[i + 1] / err[i])
            return float(np.exp(np.log(rho[i]) + w * np.log(rho[i + 1] / rho[i])))
    return None


def saving_table(hold):
    lines = [r"\begin{tabular}{rrrrr}", r"\toprule",
             r"grid & target $L_2$ & $\rho_{\rm std}$ & $\rho_{\rm hybrid}$ (best $N$) & "
             r"$\rho_{\rm std}/\rho_{\rm hybrid}$\\", r"\midrule"]
    for L in (9, 10):
        heads = dict(by_head(hold[L]))
        for target in (1e-2, 1e-3, 1e-4):
            std = rho_at(heads[0], target)
            hyb = [(rho_at(ds, target), N) for N, ds in heads.items() if N > 0]
            hyb = [h for h in hyb if h[0] is not None]
            if std is None or not hyb:
                continue
            r, N = min(hyb)
            lines.append(rf"${2**L}^2$ & {sci(target)} & {std:.4f} & {r:.4f} ({N}) "
                         rf"& {std / r:.1f}\\")
            print(f"saving L={L} target={target:g}: std={std:.4f} hyb={r:.4f} "
                  f"N={N} ratio={std / r:.2f}")
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


def tables(hold, rel, long):
    tex = [r"\newcommand{\TabSteady}{%", r"\begin{tabular}{rrrrrr}", r"\toprule",
           r"grid & $\kappa$ & $\langle\theta^2\rangle_{ss}$ & $\tau$ & "
           r"$\kappa\tau$ & within $1\%$ at\\", r"\midrule"]
    for L in (8, 9, 10):
        _, _, tau, within = rel[L]
        w = within[1e-2]
        tex.append(rf"${2**L}^2$ & {KAPPA[L]:g} & {VAR_SS[L]:.4f} & {tau:.0f} & "
                   rf"{KAPPA[L] * tau:.3f} & "
                   + (f"$t={w:.0f}$" if np.isfinite(w) else "--") + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}}", ""]
    for L in (9, 10):
        tex += [rf"\newcommand{{\TabHold{'Nine' if L == 9 else 'Ten'}}}{{%",
                grid_table(hold[L], (8, 12, 16, 24, 32), True) + "}", ""]
    tex += [r"\newcommand{\TabHoldSaving}{%", saving_table(hold) + "}", ""]
    tex += [r"\newcommand{\TabSpinup}{%",
            grid_table(load("FRC", 9), (8, 16, 24), False) + "}", ""]
    if long:
        tex += [r"\newcommand{\TabLong}{%", long_table(long) + "}", ""]
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(tex))
    print("->", OUT_TEX)


def summary(hold, rel):
    for L in (8, 9, 10):
        _, gap, tau, within = rel[L]
        print(f"relax L={L}: tau={tau:.1f} kappa*tau={KAPPA[L]*tau:.3f} "
              f"within1%={within[1e-2]:.0f} within0.1%={within[1e-3]:.0f} "
              f"final gap={gap[-1]:.1e} var_ss={VAR_SS[L]:.6f}")
    for L in (9, 10):
        print(f"hold L={L}: ref variance drift "
              f"{max(abs(d['ref_variance'][-1] / VAR_SS[L] - 1) for d in hold[L].values()):.1e}")
        for (N, bd), d in sorted(hold[L].items()):
            print(f"  N={N:2d} chi={bd:2d} rho={d['tt_params'][-1]/4**L:.4f} "
                  f"err={d['err_l2'][-1]:.2e} repr={d['repr_err_l2'][-1]:.2e} "
                  f"ratio={d['err_l2'][-1]/d['repr_err_l2'][-1]:7.2f} "
                  f"var={d['tt_variance'][-1]/VAR_SS[L]-1:+.1e} "
                  f"growth={growth(d):.2f} mass={abs(d['tt_mass'][-1]):.1e}")
    for L, tag in ((9, "FRC"), (10, "LONG")):
        for (N, bd), d in sorted(load(tag, L).items()):
            print(f"{tag}{L} N={N:2d} chi={bd:2d} T={float(d['T']):g} "
                  f"err={d['err_l2'][-1]:.2e} repr={d['repr_err_l2'][-1]:.2e} "
                  f"ratio={d['err_l2'][-1]/d['repr_err_l2'][-1]:7.2f} "
                  f"ref_var={d['ref_variance'][-1]/VAR_SS[L]-1:+.1e} "
                  f"tt-ref var={d['tt_variance'][-1]/d['ref_variance'][-1]-1:+.1e} "
                  f"growth={growth(d):.2f} mass={abs(d['tt_mass'][-1]):.1e}")
    # The extended runs repeat the T=200 hold runs exactly up to t=200.
    hold10 = hold[10]
    for (N, bd), d in sorted(load("LONG", 10).items()):
        h = hold10.get((N, bd))
        if h is None:
            continue
        _, i, j = np.intersect1d(np.round(d["t"], 6), np.round(h["t"], 6),
                                 return_indices=True)
        i, j = i[d["t"][i] > 0], j[d["t"][i] > 0]
        print(f"LONG10 vs HOLD10 N={N} chi={bd}: {len(i)} common samples, max rel "
              f"diff of err_l2 {np.max(np.abs(d['err_l2'][i] / h['err_l2'][j] - 1)):.1e}")


def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    hold = fig_hold()
    rel = fig_relaxation()
    long = fig_long()
    tables(hold, rel, long)
    summary(hold, rel)


if __name__ == "__main__":
    main()
