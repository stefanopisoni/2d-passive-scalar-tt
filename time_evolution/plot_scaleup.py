"""Hybrid vs standard TT under evolution, at two resolutions.

The L=8 comparison showed the hybrid ahead at every budget. The question here is
whether that holds up as resolution grows -- and, just as important, whether the
cheap representation-only measurement is a reliable stand-in for the expensive
evolution one. If it is, the resolution scaling can be pushed to grids where
evolving at a matched budget would be unaffordable (chi must grow ~2^L for fixed
rho, so cost grows ~8^L).

Reads results/decay_cmp_N*_bd*.npz (L=8) and results/decay_L9_N*_bd*.npz (L=9).
Writes figures/scaleup_comparison.png.
"""
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

N_COLOR = {0: "k", 4: "#1f77b4", 6: "#2ca02c", 8: "#d62728", 10: "#9467bd"}
N_LABEL = {0: "standard ($N{=}0$)", 4: "$N{=}4$", 6: "$N{=}6$", 8: "$N{=}8$",
           10: "$N{=}10$"}
L_MARKER = {8: "o", 9: "s"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 140, "savefig.bbox": "tight",
                     "legend.frameon": False})


def collect():
    """{(L, N): [rows by bd]} from both naming schemes, L read from the file."""
    pats = (re.compile(r"decay_cmp_N(\d+)_bd(\d+)\.npz$"),
            re.compile(r"decay_L\d+_N(\d+)_bd(\d+)\.npz$"))
    out = {}
    for fn in sorted(os.listdir(RESULTS)):
        m = next((p.match(fn) for p in pats if p.match(fn)), None)
        if not m:
            continue
        N, bd = int(m.group(1)), int(m.group(2))
        d = np.load(os.path.join(RESULTS, fn))
        L = int(d["L"])
        t = d["t"]
        late = t >= (2.0 / 3.0) * t[-1]
        ve = np.abs(d["tt_variance"] - d["ref_variance"]) / d["ref_variance"]
        out.setdefault((L, N), []).append(dict(
            L=L, N=N, bd=bd, d=d, t=t,
            rho=float(np.mean(d["tt_params"][late]) / float(d["n_cells"])),
            err=float(np.mean(d["err_l2"][late])),
            repr=float(np.mean(d["repr_err_l2"][late])),
            var_err=float(np.mean(ve[late])),
            ms=1000.0 * float(d["wall"]) / int(d["n_steps"]),
        ))
    for k in out:
        out[k].sort(key=lambda r: r["bd"])
    return out


def interp_err(rows, rhos):
    """Error at given budgets (a vertical cut through the rate-distortion curve)."""
    r = sorted(rows, key=lambda x: x["rho"])
    return np.exp(np.interp(np.log(rhos), np.log([x["rho"] for x in r]),
                            np.log([x["err"] for x in r]),
                            left=np.nan, right=np.nan))


def interp_rho(rows, errs):
    """Budget needed for given errors (a *horizontal* cut through the same curve).

    This is the number that answers "how many parameters does the head save",
    and it is much smaller than the error ratio at fixed budget, because the
    curves are steep: a 2-3x shift along rho shows up as one or two decades
    along the error axis. Quoting the vertical cut as "the advantage" overstates
    what the encoding buys.
    """
    r = sorted(rows, key=lambda x: x["err"])
    return np.exp(np.interp(np.log(errs), np.log([x["err"] for x in r]),
                            np.log([x["rho"] for x in r]),
                            left=np.nan, right=np.nan))


def main():
    os.makedirs(FIGS, exist_ok=True)
    data = collect()
    if not data:
        raise SystemExit("no runs found")
    Ls = sorted({L for (L, _) in data})

    fig = plt.figure(figsize=(11.5, 6.8))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)

    # --- rate-distortion at each resolution
    for i, L in enumerate(Ls[:2]):
        ax = fig.add_subplot(gs[0, i])
        for N in sorted({n for (ll, n) in data if ll == L}):
            r = data[(L, N)]
            ax.loglog([x["rho"] for x in r], [x["err"] for x in r], "-o", ms=4,
                      lw=1.4, color=N_COLOR[N], label=N_LABEL[N])
        ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$L_2$ error")
        ax.legend(fontsize=6.5)
        ax.set_title(rf"evolution error, $L={L}$ (${2**L}^2$)", fontsize=9)

    # --- the practical gain: parameters saved at a fixed accuracy.
    # The error ratio at fixed rho is the same gap read down a steep axis and
    # comes out 10x larger; it is printed in the table, not plotted here, so the
    # two cannot be mistaken for each other.
    ax = fig.add_subplot(gs[0, 2])
    errs = np.logspace(-3.5, -0.5, 40)
    for L, c in zip(Ls, ("#1f77b4", "#d62728")):
        Ns = sorted({n for (ll, n) in data if ll == L})
        if 0 not in Ns:
            continue
        hyb = [N for N in Ns if N > 0]
        std_r = interp_rho(data[(L, 0)], errs)
        best_r = np.nanmin(np.vstack([interp_rho(data[(L, N)], errs)
                                      for N in hyb]), axis=0)
        ax.semilogx(errs, std_r / best_r, "-", lw=1.8, color=c,
                    label=rf"$L={L}$ (${2**L}^2$)")
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.set_ylim(0, None)
    ax.set_xlabel(r"target $L_2$ error")
    ax.set_ylabel(r"$\rho_{\rm std}/\rho_{\rm hybrid}$")
    ax.legend(fontsize=7)
    ax.set_title("parameters saved at fixed accuracy", fontsize=9)

    # --- does representation predict evolution?
    ax = fig.add_subplot(gs[1, 0])
    for (L, N), rows in sorted(data.items()):
        ax.loglog([x["repr"] for x in rows], [x["err"] for x in rows],
                  L_MARKER.get(L, "^"), ms=5, color=N_COLOR[N],
                  mfc="none" if L != Ls[0] else N_COLOR[N], lw=0,
                  label=f"$L={L}$, {N_LABEL[N]}")
    lim = np.array([1e-5, 1e0])
    for k, ls in ((1, "-"), (3, "--"), (5, ":")):
        ax.plot(lim, k * lim, ls, color="0.5", lw=1)
        ax.text(lim[1], k * lim[1], f" {k}x", fontsize=6, color="0.5",
                va="center", ha="right")
    ax.set_xlim(2e-5, 0.3); ax.set_ylim(2e-5, 1.5)
    ax.set_xlabel("representation error"); ax.set_ylabel("evolution error")
    ax.legend(fontsize=5, ncol=2)
    ax.set_title("representation predicts evolution", fontsize=9)

    # --- variance, the bulk statistic
    ax = fig.add_subplot(gs[1, 1])
    for (L, N), rows in sorted(data.items()):
        ax.loglog([x["rho"] for x in rows], [x["var_err"] for x in rows],
                  "-" + L_MARKER.get(L, "^"), ms=4, lw=1.2, color=N_COLOR[N],
                  mfc="none" if L != Ls[0] else N_COLOR[N],
                  label=f"$L={L}$, {N_LABEL[N]}")
    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel(r"$|\Delta\langle\theta^2\rangle|/\langle\theta^2\rangle$")
    ax.legend(fontsize=5, ncol=2); ax.set_title("variance error", fontsize=9)

    # --- cost
    ax = fig.add_subplot(gs[1, 2])
    for (L, N), rows in sorted(data.items()):
        ax.loglog([x["ms"] for x in rows], [x["err"] for x in rows],
                  "-" + L_MARKER.get(L, "^"), ms=4, lw=1.2, color=N_COLOR[N],
                  mfc="none" if L != Ls[0] else N_COLOR[N],
                  label=f"$L={L}$, {N_LABEL[N]}")
    ax.set_xlabel("ms / step (shared cores)"); ax.set_ylabel(r"$L_2$ error")
    ax.legend(fontsize=5, ncol=2); ax.set_title("error vs wall-clock", fontsize=9)

    fig.suptitle("Hybrid vs standard TT under evolution, across resolution "
                 r"(fixed $\ell_B/\Delta x$)", fontsize=10, y=0.98)
    out = os.path.join(FIGS, "scaleup_comparison.png")
    fig.savefig(out); plt.close(fig)
    print(out)

    print(f"\n{'L':>3} {'N':>3} {'bd':>4} {'rho':>7} {'evo L2':>9} {'repr L2':>9} "
          f"{'ratio':>6} {'ms/step':>8}")
    for (L, N), rows in sorted(data.items()):
        for x in rows:
            print(f"{L:>3} {N:>3} {x['bd']:>4} {x['rho']:>7.4f} {x['err']:>9.2e} "
                  f"{x['repr']:>9.2e} {x['err']/max(x['repr'],1e-300):>6.1f} "
                  f"{x['ms']:>8.0f}")
    print("\nBUDGET SAVED at fixed accuracy (the practical number):")
    print("  err  " + "".join(f"{'L='+str(L):>10s}" for L in Ls))
    for te in (1e-1, 1e-2, 1e-3):
        row = []
        for L in Ls:
            Ns = sorted({n for (ll, n) in data if ll == L})
            if 0 not in Ns:
                row.append(np.nan); continue
            sr = interp_rho(data[(L, 0)], np.array([te]))[0]
            br = np.nanmin([interp_rho(data[(L, N)], np.array([te]))[0]
                            for N in Ns if N > 0])
            row.append(sr / br)
        print(f"  {te:.0e} " + "".join(f"{v:>10.1f}x" for v in row))

    print("\nERROR RATIO at fixed rho (same curves, steep axis -- larger number,"
          " same fact):")
    print("  rho  " + "".join(f"{'L='+str(L):>10s}" for L in Ls))
    for rr in (0.03, 0.05, 0.10, 0.20):
        row = []
        for L in Ls:
            Ns = sorted({n for (ll, n) in data if ll == L})
            if 0 not in Ns:
                row.append(np.nan); continue
            s = interp_err(data[(L, 0)], np.array([rr]))[0]
            b = np.nanmin([interp_err(data[(L, N)], np.array([rr]))[0]
                           for N in Ns if N > 0])
            row.append(s / b)
        print(f"  {rr:.2f} " + "".join(f"{v:>10.1f}x" for v in row))


if __name__ == "__main__":
    main()
