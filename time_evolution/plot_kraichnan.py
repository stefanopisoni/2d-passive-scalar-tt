"""Kraichnan runs: validation against the lattice spectrum, and hybrid vs standard.

Validation (decay_KV*.npz, dense): the time-averaged spectrum against the
stationary solution of the closed lattice equation (kraichnan.lattice_spectrum),
which is exact for the discrete scheme in the white-noise limit. The remaining
differences are the finite Kubo number, the time stepping and sampling noise;
the Ku and CFL variants at 256^2 separate them.

Hold runs (decay_KH*.npz): hybrid runs started from the stationary dense field,
with errors averaged over the second half of the run since the state is
stochastic.

    PSCALAR_RESULTS=<results/kraichnan> python plot_kraichnan.py

Writes figures/kraichnan_validation.png, figures/kraichnan_hold.png and
tables_kraichnan.tex; theory spectra are cached in PSCALAR_RESULTS.
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
import kraichnan as kr  # noqa: E402

RESULTS = os.environ.get("PSCALAR_RESULTS", os.path.join(_HERE, "results", "kraichnan"))
FIGS = os.environ.get("PSCALAR_FIGS", os.path.join(_HERE, "figures"))
OUT_TEX = os.path.join(os.environ.get("PSCALAR_FIGS", _HERE), "tables_kraichnan.tex")
N_COLOR = {0: "#1f77b4", 6: "#9467bd", 8: "#ff7f0e", 10: "#2ca02c", 12: "#d62728"}
XI_NAME = {"2": r"$\xi=2$ (Batchelor)", "43": r"$\xi=4/3$"}

plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight", "legend.fontsize": 7})


def xitag(xi):
    return "2" if xi >= 2 else "43"


def model_of(d):
    return kr.Kraichnan(int(d["L"]), float(d["model_xi"]), float(d["model_qmax"]),
                        float(d["model_ku"]))


def theory(d):
    """Stationary lattice spectrum for this run's (L, xi, qmax, kappa), cached.
    It does not depend on Ku: the white-noise limit sees only D_ij."""
    L, xi, kappa = int(d["L"]), float(d["model_xi"]), float(d["kappa"])
    path = os.path.join(RESULTS, f"theory_L{L}_xi{xitag(xi)}.npz")
    if os.path.exists(path):
        t = np.load(path)
        if np.isclose(float(t["kappa"]), kappa, rtol=1e-12):
            return t["C"]
    C, info = kr.lattice_spectrum(model_of(d), kappa)
    print(f"theory L={L} xi={xi:.3f}: {info}")
    np.savez(path, C=C, kappa=kappa, **{k: v for k, v in info.items()})
    return C


def load(pattern):
    out = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, pattern))):
        out[os.path.basename(f)[6:-4]] = np.load(f)
    return out


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------
def validation():
    runs = load("decay_KV*.npz")
    if not runs:
        return
    fig, axs = plt.subplots(2, 3, figsize=(11, 6))
    style = {"ku0.5": ("o", "-"), "ku0.25": ("s", "--"), "ku0.5_cfl0.25": ("^", ":")}
    color = {8: "#1f77b4", 9: "#d62728"}
    print("\nvalidation: run, var(meas)/var(theory), median |E/E_th - 1| over the "
          "resolved shells, max |..|, substeps/interval, ms/substep, intrinsic chi")
    for lab, d in runs.items():
        L, xi = int(d["L"]), float(d["model_xi"])
        row = 0 if xi >= 2 else 1
        C = theory(d)
        k, Eth = kr.shell_average(C)
        _, Em = kr.shell_average(d["spec_ref"])
        var_th = C.sum()
        late = d["t"] >= float(d["t_avg"])
        var_m = float(np.mean(d["ref_variance"][late]))
        # resolved shells: above the forcing shell's lower edge, where the theory
        # still holds 1e-6 of its peak
        good = (k >= 1) & (Eth > 1e-6 * Eth.max())
        ratio = Em[good] / Eth[good]
        variant = re.sub(r"^KV\d+_xi\w+?_", "", lab)
        mk, ls = style.get(variant, ("x", "-"))
        c = color[L]
        a0, a1, a2 = axs[row]
        if variant == "ku0.5":
            a0.loglog(k[1:], Eth[1:], color=c, lw=1.4, label=f"lattice theory {2**L}$^2$")
        a0.loglog(k[good], Em[good], mk, color=c, ms=2.5, mfc="none",
                  label=f"{2**L}$^2$ {variant}")
        a1.semilogx(k[good], ratio, mk + ls, color=c, ms=2.5, lw=0.8, mfc="none",
                    label=f"{2**L}$^2$ {variant}")
        a2.plot(d["t"], d["ref_variance"], ls=ls, color=c, lw=0.8,
                label=f"{2**L}$^2$ {variant}")
        a2.axhline(var_th, color=c, lw=0.6, ls="--")
        chi = {e: np.mean(d[f"intrinsic_bonds_{e}"][late].max(axis=1))
               for e in ("1e-06", "0.0001", "0.01")}
        print(f"  {lab:26s} {var_m / var_th:6.3f}  {np.median(np.abs(ratio - 1)):.3f}  "
              f"{np.abs(ratio - 1).max():.3f}  {d['n_sub_total'] / d['n_int']:5.1f}  "
              f"{1000 * float(np.atleast_1d(d['wall'])[-1]) / d['n_sub_total']:6.2f}  "
              + " ".join(f"chi({e})={v:.0f}" for e, v in chi.items()))
    for row, tag in enumerate(("2", "43")):
        a0, a1, a2 = axs[row]
        xi = 2.0 if tag == "2" else 4 / 3
        kk = np.array([4.0, 40.0])
        a0.loglog(kk, 0.3 * kk ** (xi - 3) * 4 ** (3 - xi) * 1e-2, "k:", lw=0.8,
                  label=rf"$k^{{{xi - 3:.2f}}}$")
        a0.set(xlabel="k", ylabel=r"$E_\theta(k)$", title=XI_NAME[tag])
        a1.axhline(1, color="k", lw=0.6)
        a1.set(xlabel="k", ylabel="measured / lattice theory", ylim=(0.7, 1.3),
               title="ratio to the white-noise limit")
        a2.set(xlabel="t", ylabel=r"$\langle\theta^2\rangle$",
               title="variance (dashed: theory)")
        for a in (a0, a1, a2):
            a.legend(loc="best")
    fig.tight_layout()
    os.makedirs(FIGS, exist_ok=True)
    out = os.path.join(FIGS, "kraichnan_validation.png")
    fig.savefig(out)
    print("->", out)


# --------------------------------------------------------------------------
# hold runs
# --------------------------------------------------------------------------
def summarise(d):
    """Errors averaged over the second half of the run."""
    late = d["t"] >= 0.5 * d["t"][-1]
    return dict(rho=float(np.mean(d["tt_params"][late])) / float(d["n_cells"]),
                err=float(np.mean(d["err_l2"][late])),
                rep=float(np.mean(d["repr_err_l2"][late])),
                var=float(np.mean(np.abs(d["tt_variance"][late] / d["ref_variance"][late] - 1))),
                growth=float(np.mean(d["err_l2"][d["t"] >= 0.75 * d["t"][-1]])
                             / np.mean(d["err_l2"][(d["t"] >= 0.5 * d["t"][-1])
                                                   & (d["t"] < 0.75 * d["t"][-1])])))


def rho_at(rows, target):
    rho = np.array([r["rho"] for r in rows])
    err = np.array([r["err"] for r in rows])
    for i in range(len(rows) - 1):
        if err[i] >= target >= err[i + 1]:
            w = np.log(target / err[i]) / np.log(err[i + 1] / err[i])
            return float(np.exp(np.log(rho[i]) + w * np.log(rho[i + 1] / rho[i])))
    return None


def hold():
    runs = load("decay_KH*.npz")
    if not runs:
        return
    groups = {}
    for lab, d in runs.items():
        m = re.match(r"KH(\d+)_xi(\w+?)_N(\d+)_bd(\d+)$", lab)
        L, tag, N, bd = int(m[1]), m[2], int(m[3]), int(m[4])
        groups.setdefault((L, tag), {}).setdefault(N, []).append((bd, d))
    keys = sorted(groups)
    fig, axs = plt.subplots(len(keys), 3, figsize=(11, 3.2 * len(keys)), squeeze=False)
    tex, saving = [], []
    for row, (L, tag) in enumerate(keys):
        a0, a1, a2 = axs[row]
        print(f"\nhold {2**L}^2 {tag}:")
        cells = {}
        for N in sorted(groups[(L, tag)]):
            rs = []
            for bd, d in sorted(groups[(L, tag)][N]):
                s = summarise(d)
                s["bd"] = bd
                rs.append(s)
                cells[(N, bd)] = s
                print(f"  N={N:2d} chi={bd:3d} rho={s['rho']:.4f} err={s['err']:.2e} "
                      f"repr={s['rep']:.2e} ratio={s['err'] / s['rep']:7.2f} "
                      f"var={s['var']:.1e} growth={s['growth']:.2f}")
                if bd == 16:
                    a2.semilogy(d["t"], d["err_l2"], color=N_COLOR[N], label=f"N={N}")
                    a2.semilogy(d["t"], d["repr_err_l2"], color=N_COLOR[N], ls="--", lw=0.8)
            a0.loglog([r["rho"] for r in rs], [r["err"] for r in rs], "o-",
                      color=N_COLOR[N], ms=4, label=f"N={N}")
            a0.loglog([r["rho"] for r in rs], [r["rep"] for r in rs], "s--",
                      color=N_COLOR[N], ms=3, lw=0.8, mfc="none")
            a1.loglog([r["rho"] for r in rs], [r["var"] for r in rs], "o-",
                      color=N_COLOR[N], ms=4, label=f"N={N}")
            groups[(L, tag)][N] = rs
        for target in (1e-1, 3e-2, 1e-2, 3e-3, 1e-3):
            std = rho_at(groups[(L, tag)].get(0, []), target)
            hyb = [(rho_at(rs, target), N) for N, rs in groups[(L, tag)].items() if N > 0]
            hyb = [h for h in hyb if h[0] is not None]
            if std is None or not hyb:
                continue
            r, N = min(hyb)
            saving.append((L, tag, target, std, r, N))
            print(f"  saving at L2={target:g}: std={std:.4f} hybrid={r:.4f} (N={N}) "
                  f"-> {std / r:.2f}")
        chis = sorted({bd for (_, bd) in cells})
        lines = [r"\begin{tabular}{r" + "c" * len(chis) + "}", r"\toprule",
                 r"$N$ & " + " & ".join(rf"$\chi={c}$" for c in chis) + r"\\", r"\midrule"]
        for N in sorted({n for (n, _) in cells}):
            cs = []
            for c in chis:
                s = cells.get((N, c))
                cs.append("--" if s is None else
                          f"{sci(s['err'])} ({s['err'] / s['rep']:.1f})")
            lines.append(f"{N} & " + " & ".join(cs) + r"\\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        tex += [rf"\newcommand{{\TabKraichnan{'Eight' if L == 8 else 'Nine'}"
                rf"{'Two' if tag == '2' else 'FourThirds'}}}{{%", "\n".join(lines) + "}", ""]
        a0.set(xlabel=r"$\rho$", ylabel=r"$L_2$ error (second half)",
               title=f"{2**L}$^2$, {XI_NAME[tag]}: evolved (o), representation (s)")
        a1.set(xlabel=r"$\rho$", ylabel=r"$|\langle\theta^2\rangle_{TT}/\langle\theta^2\rangle-1|$",
               title="variance error")
        a2.set(xlabel="t", ylabel=r"$L_2$ error", title=r"$\chi=16$: evolved (—), representation (--)")
        for a in (a0, a1, a2):
            a.legend(loc="best")
    lines = [r"\begin{tabular}{rrrrrr}", r"\toprule",
             r"grid & $\xi$ & target $L_2$ & $\rho_{\rm std}$ & $\rho_{\rm hybrid}$ ($N$) & ratio\\",
             r"\midrule"]
    for L, tag, target, std, r, N in saving:
        lines.append(rf"${2**L}^2$ & {'2' if tag == '2' else '4/3'} & {sci(target)} & "
                     rf"{std:.4f} & {r:.4f} ({N}) & {std / r:.1f}\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    tex += [r"\newcommand{\TabKraichnanSaving}{%", "\n".join(lines) + "}", ""]
    fig.tight_layout()
    out = os.path.join(FIGS, "kraichnan_hold.png")
    fig.savefig(out)
    print("->", out)
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(tex))
    print("->", OUT_TEX)


def sci(x):
    m, e = f"{x:.1e}".split("e")
    return rf"${m}\times10^{{{int(e)}}}$"


# --------------------------------------------------------------------------
# hold runs: the paper's statistics
# --------------------------------------------------------------------------
def stats(d):
    """Second-half averages of the increment statistics, spectra and budgets,
    for the TT field and the dense reference of the same run."""
    late = d["t"] >= 0.5 * d["t"][-1]
    out = dict(rho=float(np.mean(d["tt_params"][late])) / float(d["n_cells"]),
               r=np.asarray(d["r_sf"]), r_pdf=np.asarray(d["r_pdf"]),
               edges=np.asarray(d["pdf_edges"]))
    for who in ("ref", "tt"):
        sf = d[f"{who}_sf"][late].mean(0)                    # S_2, S_4, S_6 vs r
        pdf = d[f"{who}_pdf"][late].sum(0).astype(float)
        pdf /= pdf.sum(1, keepdims=True) * np.diff(out["edges"])[None, :]
        out[who] = dict(S2=sf[0], F=sf[1] / sf[0] ** 2, pdf=pdf,
                        E=d[f"{who}_E"][late].mean(0),
                        var=float(np.mean(d[f"{who}_variance"][late])),
                        chi=float(np.mean(d[f"{who}_chi"][late])))
    return out


def at_r(s, key, r):
    return s["tt"][key][list(s["r"]).index(r)] / s["ref"][key][list(s["r"]).index(r)]


def hold_stats():
    runs = load("decay_KH*.npz")
    runs = {k: d for k, d in runs.items() if "tt_sf" in d}
    if not runs:
        return
    groups = {}
    for lab, d in runs.items():
        m = re.match(r"KH(\d+)_xi(\w+?)_N(\d+)_bd(\d+)$", lab)
        groups.setdefault((int(m[1]), m[2]), {}).setdefault(int(m[3]), []).append(
            (int(m[4]), stats(d)))
    keys = sorted(groups)
    fig, axs = plt.subplots(len(keys), 5, figsize=(17, 3.2 * len(keys)), squeeze=False)
    print("\nstatistics, second-half averages, TT / reference: "
          "F(r) = S4/S2^2 at r = 2, 4, 8; S2(2); variance; dissipation kappa<|grad|^2>")
    for row, (L, tag) in enumerate(keys):
        g = {N: sorted(v, key=lambda x: x[0]) for N, v in groups[(L, tag)].items()}
        print(f"\n{2**L}^2 {tag}:")
        for N, rs in sorted(g.items()):
            for bd, s in rs:
                print(f"  N={N:2d} chi={bd:3d} rho={s['rho']:.4f}  F: "
                      + " ".join(f"{at_r(s, 'F', r):.3f}" for r in (2, 4, 8))
                      + f"  S2(2): {at_r(s, 'S2', 2):.3f}"
                      f"  var: {s['tt']['var'] / s['ref']['var']:.3f}"
                      f"  diss: {s['tt']['chi'] / s['ref']['chi']:.3f}")
            rho = [s["rho"] for _, s in rs]
            for a, (key, r) in zip(axs[row][:3], (("F", 2), ("F", 8), ("S2", 2))):
                a.semilogx(rho, [at_r(s, key, r) for _, s in rs], "o-",
                           color=N_COLOR[N], ms=4, label=f"N={N}")
        # spectra and PDFs at one budget: each N at the run closest to the rho
        # of the standard encoding at chi = 32
        ref0 = dict(g.get(0, []))
        rho_star = ref0[32]["rho"] if 32 in ref0 else np.median(
            [s["rho"] for rs in g.values() for _, s in rs])
        a3, a4 = axs[row][3], axs[row][4]
        j = list(next(iter(g.values()))[0][1]["r_pdf"]).index(2)
        drawn_ref = False
        for N, rs in sorted(g.items()):
            bd, s = min(rs, key=lambda x: abs(np.log(x[1]["rho"] / rho_star)))
            centres = 0.5 * (s["edges"][1:] + s["edges"][:-1])
            if not drawn_ref:
                a3.semilogy(centres, s["ref"]["pdf"][j], color="k", lw=1.6, label="reference")
                drawn_ref = True
            a3.semilogy(centres, s["tt"]["pdf"][j], color=N_COLOR[N], lw=0.9,
                        label=rf"N={N} $\chi$={bd} $\rho$={s['rho']:.3f}")
            k = np.arange(len(s["ref"]["E"]))
            if N == min(g):
                a4.loglog(k[1:], s["ref"]["E"][1:], color="k", lw=1.6, label="reference")
            a4.loglog(k[1:], s["tt"]["E"][1:], color=N_COLOR[N], lw=0.9,
                      label=rf"N={N} $\chi$={bd}")
        title = f"{2**L}$^2$, {XI_NAME[tag]}"
        for a, lab in zip(axs[row][:3], (r"$F_{TT}/F_{ref}$ at $r=2$",
                                         r"$F_{TT}/F_{ref}$ at $r=8$",
                                         r"$S_2^{TT}/S_2^{ref}$ at $r=2$")):
            a.axhline(1, color="k", lw=0.6)
            a.set(xlabel=r"$\rho$", ylabel=lab, title=title)
        a3.set(xlabel=r"$\delta_2\theta/\sigma_2$", ylabel="PDF", ylim=(1e-7, 1),
               xlim=(-20, 20), title=rf"increment PDF, $r=2$, $\rho\approx{rho_star:.3f}$")
        a4.set(xlabel="k", ylabel=r"$E_\theta(k)$", title=rf"spectrum, $\rho\approx{rho_star:.3f}$")
        for a in axs[row]:
            a.legend(loc="best", fontsize=6)
    fig.tight_layout()
    out = os.path.join(FIGS, "kraichnan_stats.png")
    fig.savefig(out)
    print("->", out)


# --------------------------------------------------------------------------
# compression rule: budget (KH) against plain truncation (KP), 256^2
# --------------------------------------------------------------------------
def policy_compare():
    """Same ladders under the two rules: pathwise error, variance, flatness at
    r = 2 and the dissipation, each as TT / reference, against rho."""
    kh = {k: d for k, d in load("decay_KH8_*.npz").items() if "tt_sf" in d}
    kp = {k: d for k, d in load("decay_KP8_*.npz").items() if "tt_sf" in d}
    if not kh or not kp:
        return
    tags = sorted({re.match(r"K[HP]8_xi(\w+?)_N", k)[1] for k in list(kh) + list(kp)})
    fig, axs = plt.subplots(len(tags), 4, figsize=(14, 3.2 * len(tags)), squeeze=False)
    print("\ncompression rule at 256^2, second-half averages (budget | plain):")
    for row, tag in enumerate(tags):
        for runs, ls, mk, name in ((kh, "-", "o", "budget"), (kp, "--", "s", "plain")):
            by_n = {}
            for lab, d in runs.items():
                m = re.match(r"K[HP]8_xi(\w+?)_N(\d+)_bd(\d+)$", lab)
                if m[1] != tag:
                    continue
                by_n.setdefault(int(m[2]), []).append((int(m[3]), stats(d), summarise(d)))
            for N, rs in sorted(by_n.items()):
                rs.sort(key=lambda x: x[0])
                rho = [s["rho"] for _, s, _ in rs]
                ys = ([u["err"] for _, _, u in rs],
                      [s["tt"]["var"] / s["ref"]["var"] for _, s, _ in rs],
                      [at_r(s, "F", 2) for _, s, _ in rs],
                      [s["tt"]["chi"] / s["ref"]["chi"] for _, s, _ in rs])
                for a, y in zip(axs[row], ys):
                    a.semilogx(rho, y, ls=ls, marker=mk, color=N_COLOR[N], ms=3.5,
                               mfc="none" if name == "plain" else None,
                               label=f"N={N} {name}")
                for (bd, s, u) in rs:
                    print(f"  xi={tag:2s} N={N:2d} chi={bd:3d} {name:6s} rho={s['rho']:.4f} "
                          f"err={u['err']:.2e} var={s['tt']['var'] / s['ref']['var']:.3f} "
                          f"F(2)={at_r(s, 'F', 2):.3f} diss={s['tt']['chi'] / s['ref']['chi']:.3f}")
        axs[row][0].set_yscale("log")
        for a, lab in zip(axs[row], (r"$L_2$ error", r"variance TT / ref",
                                     r"$F_{TT}/F_{ref}$, $r=2$", r"dissipation TT / ref")):
            a.set(xlabel=r"$\rho$", ylabel=lab, title=f"$256^2$, {XI_NAME[tag]}")
            if lab != r"$L_2$ error":
                a.axhline(1, color="k", lw=0.6)
        axs[row][0].legend(loc="best", fontsize=5.5, ncol=2)
    fig.tight_layout()
    out = os.path.join(FIGS, "kraichnan_policy.png")
    fig.savefig(out)
    print("->", out)


if __name__ == "__main__":
    validation()
    hold()
    hold_stats()
    policy_compare()
