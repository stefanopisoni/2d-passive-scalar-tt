"""Figures for the decaying cellular-flow cascade.

Reads the .npz files written by run_decaying_cellular.py and produces

  decay_dynamics.png   what the flow does: snapshots, spectra, variance budget,
                       where the variance sits, gradient intermittency
  decay_ranks.png      TT rank of the true solution over time, against the
                       geometric bound, at three accuracies; cost; kappa
                       dependence; and the rank the solver actually carries
  decay_accuracy.png   solver fidelity at each rank cap: error, variance,
                       spectra, mass, seam, numerical vs physical dissipation

Run run_decaying_cellular.py first.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
import hybrid_evolve as hev

RESULTS = os.path.join(_HERE, "results")
FIGS = os.path.join(_HERE, "figures")

# ---- hand-tunable ---------------------------------------------------------
SPEC_TIMES = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0)
RUNS = ("adaptive", "bd64", "bd32", "bd16")
RUN_LABEL = {"adaptive": r"adaptive $\epsilon=10^{-6}$", "bd64": r"$\chi\leq64$",
             "bd32": r"$\chi\leq32$", "bd16": r"$\chi\leq16$"}
RUN_COLOR = {"adaptive": "k", "bd64": "#1f77b4", "bd32": "#ff7f0e",
             "bd16": "#d62728"}
KAPPA_SWEEP = ("2e-3", None, "1e-2", "2e-2")   # None = the primary run
EPS_COLOR = {"1e-06": "#1f77b4", "0.0001": "#ff7f0e", "0.01": "#2ca02c"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 140, "savefig.bbox": "tight",
                     "legend.frameon": False})


def load(label):
    p = os.path.join(RESULTS, f"decay_{label}.npz")
    return np.load(p) if os.path.exists(p) else None


def nearest(t, want):
    return [int(np.argmin(np.abs(t - w))) for w in want]


def eps_tag(e):
    return f"{e:g}"


# --------------------------------------------------------------------------
def fig_dynamics(d):
    t = d["t"]
    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(3, 5, height_ratios=[1.0, 1.15, 1.15],
                          hspace=0.45, wspace=0.5)

    # --- row 1: snapshots of the true field (every one that was stored)
    vmax = np.abs(d["snaps"][0]).max()
    for j, i in enumerate(range(len(d["snap_t"]))):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(d["snaps"][i], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  origin="lower", interpolation="nearest")
        ax.set_title(rf"$t={d['snap_t'][i]:.0f}$", pad=3)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
        if j == 0:
            ax.set_ylabel(r"$\theta$", labelpad=2)

    # --- variance spectrum over time
    ax = fig.add_subplot(gs[1, :2])
    k = np.arange(d["ref_E"].shape[1])
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(SPEC_TIMES)))
    for c, i in zip(cm, nearest(t, SPEC_TIMES)):
        E = d["ref_E"][i]
        m = (k > 0) & (E > 0)
        ax.loglog(k[m], E[m], color=c, lw=1.2, label=rf"$t={t[i]:.0f}$")
    nyq = 2 ** (int(d["L"]) - 1)
    ax.set_xlabel("$k$"); ax.set_ylabel(r"$E_\theta(k)$")
    ax.set_ylim(1e-12, None)
    ax.axvline(nyq, color="0.4", ls=":", lw=1)
    ax.text(nyq * 0.88, 3e-11, "Nyquist", rotation=90, ha="right", va="bottom",
            fontsize=7, color="0.4")
    ax.legend(ncol=2, fontsize=7, loc="lower left")
    ax.set_title("variance spectrum", fontsize=9)

    # --- variance budget, with its own closure check on the same axes:
    # chi and -d/dt<theta^2>/2 lying on top of each other IS the check, so
    # there is no reason to spend a second panel on it
    ax = fig.add_subplot(gs[1, 2:4])
    ax.plot(t, d["ref_variance"], "k-", lw=1.6, label=r"$\langle\theta^2\rangle$")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\langle\theta^2\rangle$")
    ax2 = ax.twinx()
    ax2.plot(t, d["ref_chi"], color="#d62728", lw=1.4,
             label=r"$\chi=\kappa\langle|\nabla\theta|^2\rangle$")
    ax2.plot(t, -np.gradient(d["ref_variance"], t) / 2, "k--", lw=1.1,
             label=r"$-\frac{d}{dt}\langle\theta^2\rangle/2$")
    ax2.set_ylabel(r"$\chi$", color="#d62728", labelpad=1)
    ax2.tick_params(axis="y", colors="#d62728", labelsize=7, pad=1)
    ax2.locator_params(axis="y", nbins=4)
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=7, loc="upper right")
    ax.set_title("decay, dissipation, and a closed budget", fontsize=9)

    # --- how the decay depends on kappa
    ax = fig.add_subplot(gs[1, 4])
    for tag in KAPPA_SWEEP:
        dd = d if tag is None else load(f"dense_k{tag}")
        if dd is None:
            continue
        ax.semilogy(dd["t"], dd["ref_variance"], lw=1.2,
                    label=rf"$\kappa={float(dd['kappa']):g}$")
    # no y-label here: it would collide with the twinned chi axis to its left
    ax.set_xlabel("$t$")
    ax.legend(fontsize=6)
    ax.set_title(r"$\langle\theta^2\rangle$ across $\kappa$", fontsize=9)

    # --- where the variance sits
    ax = fig.add_subplot(gs[2, :2])
    ax.plot(t, d["ref_kbar"], "k-", lw=1.4, label=r"centroid $\bar k$")
    ax.plot(t, d["ref_k95"], color="#1f77b4", lw=1.2, label=r"$k_{95\%}$")
    ax.set_xlabel("$t$"); ax.set_ylabel("wavenumber")
    ax.legend(fontsize=7); ax.set_title("variance moves to small scale, then stalls",
                                        fontsize=9)

    # --- gradient intermittency
    ax = fig.add_subplot(gs[2, 2:4])
    ax.plot(t, d["ref_grad_flatness"], "k-", lw=1.4)
    ax.axhline(3.0, color="0.5", ls=":", lw=1)
    ax.text(0.98, 0.06, "Gaussian", fontsize=7, color="0.5", ha="right",
            transform=ax.transAxes)
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\langle\partial_x\theta^4\rangle/\langle\partial_x\theta^2\rangle^2$")
    ax.set_title("filaments make gradients intermittent", fontsize=9)

    # --- spectral slope in the developed range
    ax = fig.add_subplot(gs[2, 4])
    sl = []
    for i in range(len(t)):
        E = d["ref_E"][i]
        # only once the band actually carries variance; before that the fit is
        # measuring the initial condition's sharp cutoff, not a cascade
        m = (k >= 3) & (k <= 20) & (E > E.max() * 1e-10)
        sl.append(np.polyfit(np.log(k[m]), np.log(E[m]), 1)[0]
                  if m.sum() > 10 else np.nan)
    ax.plot(t, sl, "k-", lw=1.4)
    ax.axhline(-1, color="0.5", ls=":", lw=1)
    ax.text(0.98, 0.93, "Batchelor $k^{-1}$", fontsize=7, color="0.5",
            ha="right", transform=ax.transAxes)
    ax.set_ylim(-8, 0)
    ax.set_xlabel("$t$"); ax.set_ylabel("slope, $3\\leq k\\leq 20$")
    ax.set_title("steep: closed streamlines", fontsize=9)

    fig.suptitle(rf"Decaying passive scalar in a steady cellular flow "
                 rf"($2^{{{int(d['L'])}}}\times2^{{{int(d['L'])}}}$, "
                 rf"$\kappa={float(d['kappa']):g}$, skew SSPRK3) — dense ground truth",
                 fontsize=10, y=0.98)
    out = os.path.join(FIGS, "decay_dynamics.png")
    fig.savefig(out); plt.close(fig)
    return out


# --------------------------------------------------------------------------
def fig_ranks(d, runs):
    t = d["t"]
    bound = d["bond_bound"]
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.3)

    # --- max intrinsic bond vs time, per accuracy
    ax = fig.add_subplot(gs[0, 0])
    for e in d["eps_probe"]:
        tag = eps_tag(e)
        b = d[f"intrinsic_bonds_{tag}"].max(axis=1)
        ax.plot(t, b, lw=1.4, color=EPS_COLOR[tag], label=rf"$\epsilon={e:g}$")
    ax.axhline(bound.max(), color="0.4", ls="--", lw=1)
    ax.text(t[-1], bound.max(), " geometric bound", fontsize=7, color="0.4",
            va="bottom", ha="right")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\max_j \chi_j$")
    ax.legend(fontsize=7); ax.set_title("intrinsic rank saturates", fontsize=9)

    # --- bond profile at several times vs the geometric bound
    ax = fig.add_subplot(gs[0, 1])
    cut = np.arange(len(bound))
    ax.semilogy(cut, bound, "k--", lw=1.2, label="geometric bound")
    cm = plt.cm.viridis(np.linspace(0, 0.9, len(SPEC_TIMES)))
    for c, i in zip(cm, nearest(t, SPEC_TIMES)):
        ax.semilogy(cut, d["intrinsic_bonds_1e-06"][i], "-o", ms=2.5, lw=1,
                    color=c, label=rf"$t={t[i]:.0f}$")
    ax.set_xlabel("bond index (head bond first)"); ax.set_ylabel(r"$\chi_j$")
    ax.legend(fontsize=6, ncol=2); ax.set_title(r"profile at $\epsilon=10^{-6}$",
                                                fontsize=9)

    # --- compression ratio
    ax = fig.add_subplot(gs[0, 2])
    ncell = float(d["n_cells"])
    for e in d["eps_probe"]:
        tag = eps_tag(e)
        ax.semilogy(t, d[f"intrinsic_params_{tag}"] / ncell, lw=1.4,
                    color=EPS_COLOR[tag], label=rf"$\epsilon={e:g}$")
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.text(t[-1], 1.0, " dense", fontsize=7, color="0.4", va="bottom", ha="right")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\rho=$ params / $4^L$")
    ax.legend(fontsize=7); ax.set_title("cost of representing the truth", fontsize=9)

    # --- kappa dependence of the plateau
    ax = fig.add_subplot(gs[1, 0])
    ks, plateaus = [], []
    for tag in KAPPA_SWEEP:
        dd = d if tag is None else load(f"dense_k{tag}")
        if dd is None:
            continue
        tt = dd["t"]; late = tt >= 0.6 * tt[-1]
        ks.append(float(dd["kappa"]))
        plateaus.append([dd[f"intrinsic_bonds_{eps_tag(e)}"].max(axis=1)[late].mean()
                         for e in dd["eps_probe"]])
    ks = np.array(ks); plateaus = np.array(plateaus)
    o = np.argsort(ks)
    for j, e in enumerate(d["eps_probe"]):
        ax.loglog(ks[o], plateaus[o, j], "-o", ms=4, lw=1.3,
                  color=EPS_COLOR[eps_tag(e)], label=rf"$\epsilon={e:g}$")
    ax.set_xlabel(r"$\kappa$"); ax.set_ylabel(r"plateau $\max_j\chi_j$")
    ax.legend(fontsize=7); ax.set_title(r"rank grows slowly as $\kappa\!\downarrow$",
                                        fontsize=9)

    # --- rank the solver carries
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(t, d["intrinsic_bonds_1e-06"].max(axis=1), "-", color="0.55", lw=1.2,
            label="truth, $10^{-6}$")
    for lab, r in runs.items():
        ax.plot(r["t"], r["tt_bonds"].max(axis=1), lw=1.3,
                color=RUN_COLOR[lab], label=RUN_LABEL[lab])
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\max_j\chi_j$")
    ax.legend(fontsize=6); ax.set_title("rank carried by the solver", fontsize=9)

    # --- head bond specifically (the hybrid-specific one)
    ax = fig.add_subplot(gs[1, 2])
    ax.plot(t, d["intrinsic_bonds_1e-06"][:, 0], "-", color="0.55", lw=1.2,
            label="truth, $10^{-6}$")
    for lab, r in runs.items():
        ax.plot(r["t"], r["tt_bonds"][:, 0], lw=1.3, color=RUN_COLOR[lab],
                label=RUN_LABEL[lab])
    ax.axhline(bound[0], color="0.4", ls="--", lw=1)
    ax.text(t[-1], bound[0], rf" $2^N={bound[0]}$", fontsize=7, color="0.4",
            va="top", ha="right")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"head bond $\chi_0$")
    ax.legend(fontsize=6); ax.set_title("head bond runs out of room", fontsize=9)

    fig.suptitle("Rank of the solution and of the representation", fontsize=10,
                 y=0.97)
    out = os.path.join(FIGS, "decay_ranks.png")
    fig.savefig(out); plt.close(fig)
    return out


# --------------------------------------------------------------------------
def fig_accuracy(d, runs):
    t = d["t"]
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.38, wspace=0.32)

    # --- error vs time
    ax = fig.add_subplot(gs[0, 0])
    for lab, r in runs.items():
        ax.semilogy(r["t"], r["err_inf"], lw=1.3, color=RUN_COLOR[lab],
                    label=RUN_LABEL[lab])
        ax.semilogy(r["t"], r["err_l2"], lw=0.9, ls=":", color=RUN_COLOR[lab])
    ax.set_xlabel("$t$"); ax.set_ylabel("relative error")
    ax.legend(fontsize=6, title=r"solid $L_\infty$, dotted $L_2$",
              title_fontsize=6)
    ax.set_title("error saturates, does not drift", fontsize=9)

    # --- variance: does the bulk statistic survive?
    ax = fig.add_subplot(gs[0, 1])
    ax.plot(t, d["ref_variance"], "-", color="0.55", lw=2.2, label="truth")
    for lab, r in runs.items():
        ax.plot(r["t"], r["tt_variance"], lw=1.2, color=RUN_COLOR[lab],
                label=RUN_LABEL[lab])
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\langle\theta^2\rangle$")
    ax.legend(fontsize=6); ax.set_title("variance tracks even when starved",
                                        fontsize=9)

    # --- relative variance error
    ax = fig.add_subplot(gs[0, 2])
    for lab, r in runs.items():
        ax.semilogy(r["t"], np.abs(r["tt_variance"] - d["ref_variance"])
                    / d["ref_variance"], lw=1.3, color=RUN_COLOR[lab],
                    label=RUN_LABEL[lab])
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$|\Delta\langle\theta^2\rangle|/\langle\theta^2\rangle$")
    ax.legend(fontsize=6); ax.set_title("relative variance error", fontsize=9)

    # --- final spectra: where does truncation do its damage?
    ax = fig.add_subplot(gs[1, 0])
    k = np.arange(d["ref_E"].shape[1])
    E0 = d["ref_E"][-1]; m = (k > 0) & (E0 > 0)
    ax.loglog(k[m], E0[m], "-", color="0.55", lw=2.2, label="truth")
    for lab, r in runs.items():
        E = r["tt_E"][-1]; mm = (k > 0) & (E > 0)
        ax.loglog(k[mm], E[mm], lw=1.1, color=RUN_COLOR[lab], label=RUN_LABEL[lab])
    ax.set_xlabel("$k$"); ax.set_ylabel(r"$E_\theta(k)$")
    ax.set_ylim(1e-12, None); ax.legend(fontsize=6)
    ax.set_title(rf"spectrum at $t={t[-1]:.0f}$", fontsize=9)

    # --- mass conservation
    ax = fig.add_subplot(gs[1, 1])
    for lab, r in runs.items():
        ax.semilogy(r["t"], np.abs(r["tt_mass"]) + 1e-20, lw=1.3,
                    color=RUN_COLOR[lab], label=RUN_LABEL[lab])
    ax.semilogy(t, np.abs(d["ref_mass"]) + 1e-20, "-", color="0.55", lw=1.6,
                label="truth")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$|\langle\theta\rangle|$")
    ax.legend(fontsize=6); ax.set_title("mass drift", fontsize=9)

    # --- numerical vs physical dissipation
    ax = fig.add_subplot(gs[1, 2])
    for lab, r in runs.items():
        num = hev.numerical_dissipation(r["t"], r["tt_variance"],
                                        d["ref_variance"])
        ax.semilogy(r["t"], np.abs(num) / d["ref_chi"], lw=1.3,
                    color=RUN_COLOR[lab], label=RUN_LABEL[lab])
    ax.axhline(1.0, color="0.4", ls="--", lw=1)
    ax.text(t[-1], 1.0, " = molecular", fontsize=7, color="0.4", va="bottom",
            ha="right")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$|\varepsilon_{\rm num}|/\chi$")
    ax.legend(fontsize=6); ax.set_title("truncation vs molecular dissipation",
                                        fontsize=9)

    fig.suptitle("Solver fidelity at each rank cap, against the dense reference",
                 fontsize=10, y=0.97)
    out = os.path.join(FIGS, "decay_accuracy.png")
    fig.savefig(out); plt.close(fig)
    return out


def fig_fields(d, runs):
    """What rank starvation actually looks like, side by side with the truth."""
    show = ["truth"] + [l for l in RUNS if l in runs]
    nt = len(d["snap_t"])
    fig, axes = plt.subplots(len(show), nt, figsize=(1.75 * nt, 1.85 * len(show)),
                             squeeze=False)
    vmax = np.abs(d["snaps"][0]).max()
    for i, lab in enumerate(show):
        src = d if lab == "truth" else runs[lab]
        for j in range(nt):
            ax = axes[i][j]
            ax.imshow(src["snaps"][j], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      origin="lower", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            if i == 0:
                ax.set_title(rf"$t={d['snap_t'][j]:.0f}$", fontsize=9, pad=3)
            if j == 0:
                ax.set_ylabel("truth" if lab == "truth" else RUN_LABEL[lab],
                              fontsize=8)
    fig.suptitle("The field itself, at each rank cap (same colour scale)",
                 fontsize=10, y=0.965)
    fig.subplots_adjust(top=0.92, bottom=0.01, left=0.055, right=0.995,
                        hspace=0.05, wspace=0.05)
    out = os.path.join(FIGS, "decay_fields.png")
    fig.savefig(out); plt.close(fig)
    return out


def main():
    os.makedirs(FIGS, exist_ok=True)
    d = load("dense")
    if d is None:
        raise SystemExit("run run_decaying_cellular.py --dense first")
    runs = {}
    for lab in RUNS:
        r = load(lab)
        if r is not None:
            runs[lab] = r
    missing = [l for l in RUNS if l not in runs]
    if missing:
        print(f"note: missing runs {missing}")
    print(fig_dynamics(d))
    print(fig_ranks(d, runs))
    if runs:
        print(fig_accuracy(d, runs))
        print(fig_fields(d, runs))


if __name__ == "__main__":
    main()
