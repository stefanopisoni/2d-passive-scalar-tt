"""Hybrid TT versus standard TT under time evolution, at matched cost.

The paper asks how faithfully each encoding *represents* a snapshot. This asks
whether that advantage survives *evolution*: errors that are merely represented
are re-truncated once, whereas errors made while stepping are fed back into the
dynamics and can compound.

The comparison is a rate-distortion curve rather than a pair of matched runs.
Bond caps are integers, so no two encodings land on exactly the same parameter
count; instead each encoding is swept over bond caps and plotted as error
against the compression ratio it actually achieved. Whichever curve lies lower
at a given rho wins at that budget.

N=0 is the standard MPS (verified to reproduce tt.vector_to_mps(vec, 2, 2L)
bond-for-bond, up to tensnet's renormalisation factor); N>0 are hybrids with a
2^N dense head. Everything else -- grid, velocity, stencils, time scheme, dt,
truncation policy, and the dense reference -- is identical across runs, so the
head size is the only variable.

Reads results/decay_cmp_N*_bd*.npz written by run_decaying_cellular.py.
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
N_LABEL = {0: "standard TT ($N=0$)", 4: "hybrid $N=4$", 6: "hybrid $N=6$",
           8: "hybrid $N=8$", 10: "hybrid $N=10$"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 140, "savefig.bbox": "tight",
                     "legend.frameon": False})


def collect():
    """{N: [rows sorted by bd]}, each row a dict of late-time summaries."""
    pat = re.compile(r"decay_cmp_N(\d+)_bd(\d+)\.npz$")
    out = {}
    for fn in sorted(os.listdir(RESULTS)):
        m = pat.match(fn)
        if not m:
            continue
        N, bd = int(m.group(1)), int(m.group(2))
        d = np.load(os.path.join(RESULTS, fn))
        t = d["t"]
        # average over the last third, where the error has saturated
        late = t >= (2.0 / 3.0) * t[-1]
        ve = np.abs(d["tt_variance"] - d["ref_variance"]) / d["ref_variance"]
        out.setdefault(N, []).append(dict(
            N=N, bd=bd, d=d, t=t, late=late,
            rho=float(np.mean(d["tt_params"][late]) / float(d["n_cells"])),
            err_inf=float(np.mean(d["err_inf"][late])),
            err_l2=float(np.mean(d["err_l2"][late])),
            repr_inf=float(np.mean(d["repr_err_inf"][late])),
            repr_l2=float(np.mean(d["repr_err_l2"][late])),
            var_err=float(np.mean(ve[late])),
            ms=1000.0 * float(d["wall"]) / int(d["n_steps"]),
            flat=float(np.std(d["err_l2"][late]) / max(np.mean(d["err_l2"][late]), 1e-300)),
        ))
    for N in out:
        out[N].sort(key=lambda r: r["bd"])
    return out


def main():
    os.makedirs(FIGS, exist_ok=True)
    data = collect()
    if not data:
        raise SystemExit("no comparison runs found")

    fig = plt.figure(figsize=(11, 6.6))
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.33)

    # --- the headline: evolution error vs achieved compression
    ax = fig.add_subplot(gs[0, 0])
    for N in sorted(data):
        r = data[N]
        ax.loglog([x["rho"] for x in r], [x["err_l2"] for x in r], "-o", ms=4,
                  lw=1.4, color=N_COLOR[N], label=N_LABEL[N])
    ax.set_xlabel(r"$\rho$ = params / $4^L$"); ax.set_ylabel(r"$L_2$ error")
    ax.legend(fontsize=6.5)
    ax.set_title("evolution error vs cost", fontsize=9)

    # --- the control: representation error at the same budget
    ax = fig.add_subplot(gs[0, 1])
    for N in sorted(data):
        r = data[N]
        ax.loglog([x["rho"] for x in r], [x["repr_l2"] for x in r], "--s", ms=3.5,
                  lw=1.2, color=N_COLOR[N], label=N_LABEL[N])
    ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$L_2$ error")
    ax.legend(fontsize=6.5)
    ax.set_title("representation error (control)", fontsize=9)

    # --- how much evolution costs on top of representation
    ax = fig.add_subplot(gs[0, 2])
    for N in sorted(data):
        r = data[N]
        ax.semilogx([x["rho"] for x in r],
                    [x["err_l2"] / max(x["repr_l2"], 1e-300) for x in r],
                    "-o", ms=4, lw=1.4, color=N_COLOR[N], label=N_LABEL[N])
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.text(0.98, 0.06, "no penalty", fontsize=7, color="0.4", ha="right",
            transform=ax.transAxes)
    ax.set_xlabel(r"$\rho$"); ax.set_ylabel("evolution / representation")
    ax.set_yscale("log"); ax.legend(fontsize=6.5)
    ax.set_title("penalty for evolving, not just storing", fontsize=9)

    # --- variance, the bulk statistic
    ax = fig.add_subplot(gs[1, 0])
    for N in sorted(data):
        r = data[N]
        ax.loglog([x["rho"] for x in r], [x["var_err"] for x in r], "-o", ms=4,
                  lw=1.4, color=N_COLOR[N], label=N_LABEL[N])
    ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$|\Delta\langle\theta^2\rangle|/\langle\theta^2\rangle$")
    ax.legend(fontsize=6.5); ax.set_title("variance error vs cost", fontsize=9)

    # --- error against wall-clock, the practical axis
    ax = fig.add_subplot(gs[1, 1])
    for N in sorted(data):
        r = data[N]
        ax.loglog([x["ms"] for x in r], [x["err_l2"] for x in r], "-o", ms=4,
                  lw=1.4, color=N_COLOR[N], label=N_LABEL[N])
    ax.set_xlabel("ms / step (5 runs sharing 10 cores)")
    ax.set_ylabel(r"$L_2$ error")
    ax.legend(fontsize=6.5); ax.set_title("error vs wall-clock", fontsize=9)

    # --- error histories, to show saturation rather than drift
    ax = fig.add_subplot(gs[1, 2])
    for N in sorted(data):
        for x in data[N]:
            if x["bd"] not in (12, 32):
                continue
            ax.semilogy(x["t"], x["d"]["err_l2"], lw=1.2, color=N_COLOR[N],
                        ls="-" if x["bd"] == 32 else ":",
                        label=f"{N_LABEL[x['N']]}, $\\chi\\leq{x['bd']}$")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$L_2$ error")
    ax.legend(fontsize=5.5, ncol=2); ax.set_title("saturation, not drift", fontsize=9)

    fig.suptitle("Hybrid TT vs standard TT under evolution "
                 r"(decaying cellular flow, $2^8\times2^8$, $\kappa=5\times10^{-3}$)",
                 fontsize=10, y=0.98)
    out = os.path.join(FIGS, "hybrid_vs_standard.png")
    fig.savefig(out); plt.close(fig)
    print(out)

    # --- the numbers, since the curves are close
    print(f"\n{'N':>3} {'bd':>3} {'rho':>7} {'evo L2':>9} {'repr L2':>9} "
          f"{'ratio':>6} {'var err':>9} {'ms/step':>8}")
    for N in sorted(data):
        for x in data[N]:
            print(f"{x['N']:>3} {x['bd']:>3} {x['rho']:>7.4f} {x['err_l2']:>9.2e} "
                  f"{x['repr_l2']:>9.2e} {x['err_l2']/max(x['repr_l2'],1e-300):>6.1f} "
                  f"{x['var_err']:>9.2e} {x['ms']:>8.0f}")

    # --- interpolate each encoding onto a common rho grid to call a winner
    print("\nL2 error interpolated to common rho (log-log):")
    rl = np.log(np.array([0.05, 0.10, 0.20]))
    hdr = "  rho   " + "".join(f"{N_LABEL[N].split('(')[0].strip():>22s}" for N in sorted(data))
    print(hdr)
    curves = {}
    for N in sorted(data):
        r = sorted(data[N], key=lambda x: x["rho"])
        curves[N] = np.exp(np.interp(rl, np.log([x["rho"] for x in r]),
                                     np.log([x["err_l2"] for x in r]),
                                     left=np.nan, right=np.nan))
    for i, rr in enumerate(np.exp(rl)):
        print(f"  {rr:.2f}  " + "".join(f"{curves[N][i]:>22.3e}" for N in sorted(data)))


if __name__ == "__main__":
    main()
