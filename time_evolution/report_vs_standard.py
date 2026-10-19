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


# --------------------------------------------------------------------------
# conventions tables
# --------------------------------------------------------------------------
# gamma = rms strain rate, (2 <S_ij S_ij>)^{1/2}.  In an incompressible 2-D
# field this coincides with <|grad u|^2>^{1/2} and with the rms vorticity; all
# three were measured and agree to 4 digits.  The cellular flow gives 1 exactly
# (measured 0.9999); the truncated DNS velocity at L=9 measures 14.601.
GAMMA = {"cellular": 1.0, "dns": 14.601}

# chi of the velocity, after the derivative MPO and its own truncation
CHI_U = {"cellular": 4, "dns": 34}

# eps used to encode the streamfunction
EPS_PSI = {"cellular": "$10^{-13}$", "dns": "$10^{-2}$"}


def tab_symbols():
    """Which object each of chi, eps, rho refers to, and which one is swept."""
    rows = [
        (r"$\chi$", r"largest bond dimension",
         r"\textbf{the swept control} (\code{bd\_first}~$=$~\code{bd\_remaining});"
         r" binding in every production run, i.e.\ measured $\chi=\chi_{\rm cap}$",
         r"fixed per object, never swept: $4$ (cellular $\mathbf{u}$),"
         r" $34$ (DNS $\mathbf{u}$), $2$ ($f$)"),
        (r"$\epsilon$", r"relative SVD discard threshold",
         r"\textbf{off} ($\epsilon=$\code{None}) in all $101$ production runs;"
         r" recorded as $-1$ in the run files",
         r"fixed: $10^{-13}$ ($\psi$ cellular, $f$), $10^{-2}$ ($\psi$ DNS),"
         r" $10^{-10}$ after the derivative MPO"),
        (r"$\rho$", r"stored parameters divided by $4^L$",
         r"\textbf{an output}, determined by $(N,\chi)$ --- not an input",
         r"output; $\rho(\mathbf{u})=1.1\times10^{-3}$ to $1.9\times10^{-2}$"
         r" (cellular), $0.14$ (DNS)"),
    ]
    out = [r"\begin{tabular}{llp{0.30\linewidth}p{0.30\linewidth}}", r"\toprule",
           r"& & applied to $\theta$ (the state) & applied to $\mathbf{u},\psi,f$"
           r" (the coefficients)\\", r"\midrule"]
    for s, d, a, b in rows:
        out.append(rf"{s} & {d} & \raggedright {a} & \raggedright {b}"
                   r"\tabularnewline[2pt]")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def run_families():
    """One record per run family, read off the result files themselves."""
    fams = {}
    for fn in sorted(glob.glob(os.path.join(RESULTS, "decay_*.npz"))):
        d = np.load(fn, allow_pickle=True)
        if bool(d["dense_only"]) or str(d["label"]) == "dnssmoke":
            continue
        vel = str(d["velocity"]) if "velocity" in d.files else "cellular"
        amp = float(d["forcing_amp"]) if "forcing_amp" in d.files else 0.0
        kind = "dns" if vel == "dns" else ("forced" if amp else "decaying")
        key = (kind, int(d["L"]), float(d["T"]))
        f = fams.setdefault(key, dict(
            kind=kind, flow=vel, L=int(d["L"]), T=float(d["T"]),
            kappa=float(d["kappa"]), dt=float(d["dt"]),
            steps=int(d["n_steps"]), amp=amp,
            kf=int(d["forcing_k"]) if "forcing_k" in d.files else 0,
            Ns=set(), chis=set(), eps=set(), rho=[], n=0))
        f["Ns"].add(int(d["N"]))
        f["chis"].add(int(d["bd"]))
        f["eps"].add(float(d["eps"]))
        f["rho"].append(float(d["tt_params"][-1]) / 4.0 ** int(d["L"]))
        f["n"] += 1
    return [fams[k] for k in sorted(fams, key=lambda k: (k[0], k[1], k[2]))]


def tab_runs(fams):
    """Master run table: every knob of every family, in one place."""
    def setstr(s):
        return ",\\,".join(str(x) for x in sorted(s))

    def sci(x):
        m, e = f"{x:.4e}".split("e")
        return rf"${m.rstrip('0').rstrip('.')}\times10^{{{int(e)}}}$"

    label = {"decaying": "cellular, decaying", "forced": "cellular, forced",
             "dns": "frozen DNS"}
    out = [r"\begin{tabular}{lrlrrrlllr}", r"\toprule",
           r"flow & grid & $\kappa$ & $\ell_B/\Delta x$ & $T$ & steps &"
           r" $N$ & $\chi(\theta)$ & $\rho(\theta)$ & $\chi(\mathbf{u})$\\",
           r"\midrule"]
    for f in fams:
        g = GAMMA[f["flow"]]
        lb = np.sqrt(f["kappa"] / g) / (2 * np.pi / 2 ** f["L"])
        name = label[f["kind"]]
        if f["kind"] == "forced":
            name += rf" ($f={f['amp']:g}\sin {f['kf']}y$)"
        eps_off = f["eps"] == {-1.0}
        rho = rf"{min(f['rho']):.4f}--{max(f['rho']):.4f}"
        if not eps_off:
            rho += r"$^{\dagger}$"
        out.append(
            rf"{name} & ${2**f['L']}^2$ & {sci(f['kappa'])} & {lb:.2f} & "
            rf"{f['T']:g} & {f['steps']} & {setstr(f['Ns'])} & "
            rf"{setstr(f['chis'])} & {rho} & {CHI_U[f['flow']]}\\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out)


def conventions_tables():
    fams = run_families()
    with open(os.path.join(_HERE, "tables_vs_standard.tex"), "a") as f:
        f.write("\n\\newcommand{\\TabSymbols}{%\n" + tab_symbols() + "}\n")
        f.write("\n\\newcommand{\\TabRuns}{%\n" + tab_runs(fams) + "}\n")
    return sum(x["n"] for x in fams), len(fams)


if __name__ == "__main__":
    rows = collect()
    print(fig_head_budget(rows))
    print(latex_tables(rows))
    dns = collect(r"decay_DNS9_N(\d+)_bd(\d+)\.npz$")
    frc = collect(r"decay_FRC8_N(\d+)_bd(\d+)\.npz$")
    extra_tables(dns, frc)
    n_runs, n_fams = conventions_tables()
    print(f"{len(rows)} cellular, {len(dns)} DNS, {len(frc)} forced runs tabulated")
    print(f"conventions: {n_runs} production runs in {n_fams} families")
