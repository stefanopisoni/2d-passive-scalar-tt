"""Tables and the head-budget figure for the hybrid-vs-standard TT report.

Writes figures/head_budget.png and tables_vs_standard.tex (LaTeX fragments
included by report_vs_standard.tex).
"""
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

RESULTS = os.path.join(_HERE, "results")
FIGS = os.path.join(_HERE, "figures")
N_COLOR = {0: "k", 4: "#1f77b4", 6: "#2ca02c", 8: "#d62728", 10: "#9467bd",
           12: "#8c564b"}
L_COLOR = {8: "#1f77b4", 9: "#d62728", 10: "#2ca02c", 11: "#9467bd"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "legend.frameon": False})


def collect(pattern=r"decay_(?:cmp|L\d+)_N(\d+)_bd(\d+)\.npz$"):
    rows = []
    pat = re.compile(pattern)
    for fn in sorted(glob.glob(os.path.join(RESULTS, "decay_*.npz"))):
        m = pat.search(os.path.basename(fn))
        if not m:
            continue
        d = np.load(fn)
        L, N, bd = int(d["L"]), int(m.group(1)), int(m.group(2))
        t = d["t"]
        late = t >= (2.0 / 3.0) * t[-1]
        bonds = d["tt_bonds"][-1]
        tot = int(d["tt_params"][-1])
        head = (2 ** N) * int(bonds[0]) if N > 0 else int(bonds[0])
        rows.append(dict(
            L=L, N=N, bd=bd, chi0=int(bonds[0]), rho=tot / 4.0 ** L,
            head_frac=head / tot, head_of_dense=head / 4.0 ** L,
            err=float(np.mean(d["err_l2"][late])),
            rep=float(np.mean(d["repr_err_l2"][late])),
            var=float(np.mean(np.abs(d["tt_variance"] - d["ref_variance"])
                              / d["ref_variance"])[late] if False else
                      np.mean((np.abs(d["tt_variance"] - d["ref_variance"])
                               / d["ref_variance"])[late])),
            ms=1000.0 * float(d["wall"]) / int(d["n_steps"]),
        ))
    return rows


def interp_rho(rows, target):
    """Budget needed to reach `target` error, by log-log interpolation."""
    r = sorted(rows, key=lambda x: x["err"])
    e = np.array([x["err"] for x in r])
    p = np.array([x["rho"] for x in r])
    if target < e[0] or target > e[-1]:
        return np.nan
    return float(np.exp(np.interp(np.log(target), np.log(e), np.log(p))))


def fig_head_budget(rows):
    Ls = sorted({r["L"] for r in rows})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.3))

    ax = axes[0]
    for L in Ls:
        Ns = sorted({r["N"] for r in rows if r["L"] == L and r["N"] > 0})
        frac = [np.mean([r["head_frac"] for r in rows
                         if r["L"] == L and r["N"] == N]) for N in Ns]
        ax.plot(Ns, np.array(frac) * 100, "-o", ms=4, lw=1.4,
                color=L_COLOR.get(L, "0.5"), label=rf"$L={L}$")
    ax.set_xlabel("$N$ (head legs)")
    ax.set_ylabel("head share of the compressed state (\\%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=7)
    ax.set_title("share of parameters held by the head", fontsize=9)

    ax = axes[1]
    for L in Ls:
        Ns = sorted({r["N"] for r in rows if r["L"] == L and r["N"] > 0})
        frac = [np.mean([r["head_of_dense"] for r in rows
                         if r["L"] == L and r["N"] == N]) for N in Ns]
        ax.semilogy(Ns, frac, "-o", ms=4, lw=1.4, color=L_COLOR.get(L, "0.5"),
                    label=rf"$L={L}$")
    ax.set_xlabel("$N$ (head legs)")
    ax.set_ylabel(r"head size / $4^L$")
    ax.legend(fontsize=7)
    ax.set_title("head size relative to the dense field", fontsize=9)

    ax = axes[2]
    for L in Ls:
        for N in sorted({r["N"] for r in rows if r["L"] == L}):
            sel = sorted([r for r in rows if r["L"] == L and r["N"] == N],
                         key=lambda x: x["rho"])
            ax.loglog([x["rho"] for x in sel], [x["err"] for x in sel], "-o",
                      ms=3, lw=1.0, alpha=0.9 if L == 10 else 0.25,
                      color=N_COLOR.get(N, "0.5"),
                      label=(rf"$N={N}$" if L == 10 else None))
    ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$L_2$ error")
    ax.legend(fontsize=7, title=r"$L=10$ solid", title_fontsize=7)
    ax.set_title(r"rate--distortion, all $L$ (faint), $L=10$ solid", fontsize=9)

    out = os.path.join(FIGS, "head_budget.png")
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


def latex_tables(rows):
    Ls = sorted({r["L"] for r in rows})
    out = []

    # --- head-size parametrisation
    out.append(r"\begin{tabular}{rrrrrr}")
    out.append(r"\toprule")
    out.append(r"$N$ & bits/axis & blocks & \multicolumn{3}{c}{cells per block}\\")
    out.append(r"\cmidrule(l){4-6}")
    out.append(r" & $N/2$ & $2^{N/2}\times2^{N/2}$ & $L=9$ & $L=10$ & $L=11$\\")
    out.append(r"\midrule")
    for N in (4, 6, 8, 10, 12):
        cells = " & ".join(str(2 ** (L - N // 2)) if N // 2 <= L else "--"
                           for L in (9, 10, 11))
        out.append(rf"{N} & {N//2} & ${2**(N//2)}\times{2**(N//2)}$ & {cells}\\")
    out.append(r"\bottomrule")
    out.append(r"\end{tabular}")
    tab_head = "\n".join(out)

    # --- rate-distortion per L
    rd = []
    for L in Ls:
        rd.append(rf"\textbf{{$L={L}$ (${2**L}^2$)}}\\[2pt]")
        rd.append(r"\begin{tabular}{rrrrrr}")
        rd.append(r"\toprule")
        rd.append(r"$N$ & $\chi$ & $\rho$ & head\,\% & $L_2$ (evolution) & $L_2$ (representation)\\")
        rd.append(r"\midrule")
        for r in sorted([x for x in rows if x["L"] == L],
                        key=lambda x: (x["N"], x["bd"])):
            hf = "--" if r["N"] == 0 else f"{100*r['head_frac']:.0f}"
            rd.append(rf"{r['N']} & {r['bd']} & {r['rho']:.4f} & {hf} & "
                      rf"{r['err']:.2e} & {r['rep']:.2e}\\")
        rd.append(r"\bottomrule")
        rd.append(r"\end{tabular}")
        rd.append(r"\vspace{6pt}")
    tab_rd = "\n".join(rd).replace("e-0", r"\mathrm{e}{-}").replace("e+0", r"\mathrm{e}{+}")
    tab_rd = re.sub(r"(\d\.\d\d)\\mathrm\{e\}\{-\}(\d+)", r"$\1\\times10^{-\2}$", tab_rd)

    # --- budget saving
    targets = (1e-1, 6e-2, 3e-2, 1e-2, 3e-3)
    bs = [r"\begin{tabular}{r" + "r" * len(Ls) + "}", r"\toprule",
          "target $L_2$ & " + " & ".join(rf"$L={L}$" for L in Ls) + r"\\",
          r"\midrule"]
    for te in targets:
        cells = []
        for L in Ls:
            sub = [r for r in rows if r["L"] == L]
            std = [r for r in sub if r["N"] == 0]
            hyb_by_N = {}
            for N in sorted({r["N"] for r in sub if r["N"] > 0}):
                v = interp_rho([r for r in sub if r["N"] == N], te)
                if np.isfinite(v):
                    hyb_by_N[N] = v
            s = interp_rho(std, te) if std else np.nan
            if not np.isfinite(s) or not hyb_by_N:
                cells.append("--")
            else:
                bN = min(hyb_by_N, key=hyb_by_N.get)
                cells.append(rf"{s/hyb_by_N[bN]:.1f} ($N{{=}}{bN}$)")
        mant, exp = f"{te:.0e}".split("e")
        label = rf"${int(mant)}\times10^{{{int(exp)}}}$"
        bs.append(label + " & " + " & ".join(cells) + r"\\")
    bs += [r"\bottomrule", r"\end{tabular}"]
    tab_bs = "\n".join(bs)

    # --- cost
    cs = [r"\begin{tabular}{rrrrr}", r"\toprule",
          r"$L$ & $N$ & $\chi$ & $\rho$ & ms/step\\", r"\midrule"]
    for L in Ls:
        for r in sorted([x for x in rows if x["L"] == L and x["bd"] == 32],
                        key=lambda x: x["N"]):
            cs.append(rf"{L} & {r['N']} & {r['bd']} & {r['rho']:.4f} & {r['ms']:.0f}\\")
    cs += [r"\bottomrule", r"\end{tabular}"]
    tab_cost = "\n".join(cs)

    with open(os.path.join(_HERE, "tables_vs_standard.tex"), "w") as f:
        f.write("\\newcommand{\\TabHeadSizes}{%\n" + tab_head + "}\n\n")
        f.write("\\newcommand{\\TabRateDistortion}{%\n" + tab_rd + "}\n\n")
        f.write("\\newcommand{\\TabBudgetSaving}{%\n" + tab_bs + "}\n\n")
        f.write("\\newcommand{\\TabCost}{%\n" + tab_cost + "}\n")
    return os.path.join(_HERE, "tables_vs_standard.tex")


def extra_tables(rows_dns, rows_frc):
    """Tables for the two run families that use a different flow."""
    def block(rows, caption_cols):
        out = [r"\begin{tabular}{rrrrrr}", r"\toprule",
               r"$N$ & $\chi$ & $\rho$ & head\,\% & $L_2$ (evolution) & "
               r"$L_2$ (representation)\\", r"\midrule"]
        for r in sorted(rows, key=lambda x: (x["N"], x["bd"])):
            hf = "--" if r["N"] == 0 else f"{100*r['head_frac']:.0f}"
            out.append(rf"{r['N']} & {r['bd']} & {r['rho']:.4f} & {hf} & "
                       rf"${r['err']:.2e} & ${r['rep']:.2e}\\".replace(
                           "e-0", r"\times10^{-").replace("e+0", r"\times10^{+"))
        out += [r"\bottomrule", r"\end{tabular}"]
        txt = "\n".join(out)
        return re.sub(r"\$(\d\.\d\d)\\times10\^\{-(\d+)", r"$\1\\times10^{-\2}$", txt)

    with open(os.path.join(_HERE, "tables_vs_standard.tex"), "a") as f:
        f.write("\n\\newcommand{\\TabDNS}{%\n" + block(rows_dns, None) + "}\n")
        f.write("\n\\newcommand{\\TabForced}{%\n" + block(rows_frc, None) + "}\n")


if __name__ == "__main__":
    rows = collect()
    print(fig_head_budget(rows))
    print(latex_tables(rows))
    dns = collect(r"decay_DNS9_N(\d+)_bd(\d+)\.npz$")
    frc = collect(r"decay_FRC8_N(\d+)_bd(\d+)\.npz$")
    extra_tables(dns, frc)
    print(f"{len(rows)} cellular, {len(dns)} DNS, {len(frc)} forced runs tabulated")
