"""Time histories for the report and the recap: physics and compression.

Reads the .npz files written by run_decaying_cellular.py and produces

  history_physical.png     what the flow does to the scalar over time: variance,
                           the budget of Eq. (2) closed numerically, where the
                           variance sits in k, gradient intermittency, the
                           spectrum at several times, the grid-cutoff check
  history_compression.png  what the encoding does over the same window: error
                           against budget, evolution versus representation,
                           intrinsic and carried rank, stored parameters,
                           variance error, mass drift

Both cover all three flows (cellular decaying, cellular forced, frozen DNS) so
that a statement made for one can be checked against the others. Run
run_decaying_cellular.py first.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

RESULTS = os.path.join(_HERE, "results")
FIGS = os.path.join(_HERE, "figures")

# ---- hand-tunable ---------------------------------------------------------
# The reference run of each flow: converged enough that its history is the
# physics rather than the truncation. Everything in history_physical.png is a
# dense-reference quantity, so the bond cap only has to be large enough that the
# run completed.
FLOW = {
    "decaying": dict(label="L10_N10_bd64", name=r"cellular, decaying ($1024^2$)",
                     color="#1f77b4"),
    "forced": dict(label="FRC8_N8_bd32", name=r"cellular, forced ($256^2$)",
                   color="#2ca02c"),
    "dns": dict(label="DNS9_N10_bd48", name=r"frozen DNS ($512^2$)",
                color="#d62728"),
}
# Budget caps compared in history_compression.png, at one head size.
CMP = dict(L=10, N=10, bds=(16, 24, 32, 48, 64))
BD_COLOR = {16: "#d62728", 24: "#ff7f0e", 32: "#2ca02c", 48: "#1f77b4",
            64: "#9467bd"}
EPS_COLOR = {"1e-06": "#1f77b4", "0.0001": "#ff7f0e", "0.01": "#2ca02c"}
EPS_LABEL = {"1e-06": r"$\epsilon=10^{-6}$", "0.0001": r"$\epsilon=10^{-4}$",
             "0.01": r"$\epsilon=10^{-2}$"}
SPEC_FRAC = (0.0, 0.05, 0.15, 0.4, 1.0)      # times as a fraction of T
plt.rcParams.update({"font.size": 8, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "legend.frameon": False})


def load(label):
    p = os.path.join(RESULTS, f"decay_{label}.npz")
    return np.load(p, allow_pickle=True) if os.path.exists(p) else None


def forcing_of(d):
    """The steady source as a dense field, or None. Needed for the production
    term of the budget, which is not stored."""
    amp = float(d["forcing_amp"]) if "forcing_amp" in d.files else 0.0
    if not amp:
        return None
    import run_decaying_cellular as R
    return R.forcing_field(int(d["L"]), amp, int(d["forcing_k"]))


def carried_chi(d):
    return np.array([np.max(b) for b in d["tt_bonds"]], dtype=float)


def intrinsic_chi(d, tag):
    return np.array([np.max(b) for b in d[f"intrinsic_bonds_{tag}"]], dtype=float)


# --------------------------------------------------------------------------
def fig_physical(out):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))

    # (a) variance, each flow normalised by its own initial value
    ax = axes[0, 0]
    for k, m in FLOW.items():
        d = load(m["label"])
        if d is None:
            continue
        v = d["ref_variance"]
        ax.plot(d["t"] / float(d["T"]), v / v[0], color=m["color"],
                label=m["name"])
    ax.set_xlabel("$t/T$")
    ax.set_ylabel(r"$\langle\theta^2\rangle/\langle\theta^2\rangle_0$")
    ax.legend(fontsize=6.5)
    ax.set_title("variance: decay, plateau, near-conservation", fontsize=8)

    # (b) the budget of Eq. (2), closed numerically.  The unforced flows are
    # checked at every sample; the forced one only where a full snapshot was
    # stored, since the production term <theta f> is not a recorded diagnostic.
    ax = axes[0, 1]
    for k, m in FLOW.items():
        d = load(m["label"])
        if d is None:
            continue
        t, v = d["t"], d["ref_variance"]
        lhs = -0.5 * np.gradient(v, t)              # -d<th^2>/2dt
        f = forcing_of(d)
        if f is None:
            ok = (t > t[1]) & (d["ref_chi"] > 0)
            ax.plot(t[ok] / float(d["T"]), lhs[ok] / d["ref_chi"][ok],
                    color=m["color"], label=m["name"])
        else:
            st = d["snap_t"]
            prod = np.array([float((s * f).mean()) for s in d["snaps"]])
            diss = np.interp(st, t, d["ref_chi"])
            dvdt = np.interp(st, t, lhs)
            good = st > 0
            ax.plot(st[good] / float(d["T"]),
                    (dvdt[good] + prod[good]) / diss[good], "o",
                    color=m["color"], ms=5, label=m["name"] + " (snapshots)")
    ax.axhline(1.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("$t/T$")
    ax.set_ylabel(r"$(-\frac{d}{dt}\frac{\langle\theta^2\rangle}{2}"
                  r"+\langle\theta f\rangle)\ /\ "
                  r"\kappa\langle|\nabla\theta|^2\rangle$")
    ax.set_ylim(0.6, 1.4)
    ax.legend(fontsize=6.5)
    ax.set_title("the variance budget closes", fontsize=8)

    # (c) where the variance sits
    ax = axes[0, 2]
    for k, m in FLOW.items():
        d = load(m["label"])
        if d is None:
            continue
        x = d["t"] / float(d["T"])
        ax.plot(x, d["ref_kbar"], color=m["color"], label=rf"$\bar k$, {k}")
        ax.plot(x, d["ref_k95"], color=m["color"], ls="--", lw=0.9)
    ax.set_yscale("log")
    ax.set_xlabel("$t/T$")
    ax.set_ylabel(r"$\bar k$ (solid), $k_{95}$ (dashed)")
    ax.legend(fontsize=6.5)
    ax.set_title("variance moves to small scale, then stalls", fontsize=8)

    # (d) gradient intermittency
    ax = axes[1, 0]
    for k, m in FLOW.items():
        d = load(m["label"])
        if d is None:
            continue
        ax.plot(d["t"] / float(d["T"]), d["ref_grad_flatness"],
                color=m["color"], label=m["name"])
    ax.axhline(3.0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("$t/T$")
    ax.set_ylabel(r"$\langle(\partial_x\theta)^4\rangle/"
                  r"\langle(\partial_x\theta)^2\rangle^2$")
    ax.set_title("filaments make gradients non-Gaussian (3 = Gaussian)",
                 fontsize=8)

    # (e) variance spectrum at several times, decaying run
    ax = axes[1, 1]
    d = load(FLOW["decaying"]["label"])
    if d is not None:
        t, E = d["t"], d["ref_E"]
        kk = np.arange(1, E.shape[1] + 1)
        cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(SPEC_FRAC)))
        for c, fr in zip(cmap, SPEC_FRAC):
            i = int(np.argmin(np.abs(t - fr * t[-1])))
            e = E[i]
            ax.loglog(kk[e > 0], e[e > 0], color=c, lw=1.0,
                      label=rf"$t={t[i]:.1f}$")
        ax.legend(fontsize=6, ncol=2)
        peak = E[E > 0].max()
        ax.set_ylim(peak * 1e-14, peak * 5)
    ax.set_xlabel("$k$")
    ax.set_ylabel("$E_\\theta(k)$")
    ax.set_title("spectrum fills in, then falls together", fontsize=8)

    # (f) scalar dissipation: the cascade arriving at the Batchelor scale
    ax = axes[1, 2]
    for k, m in FLOW.items():
        d = load(m["label"])
        if d is None:
            continue
        c = d["ref_chi"]
        ax.plot(d["t"] / float(d["T"]), c / c[0], color=m["color"],
                label=rf"{m['name']}, $t=0$ value {c[0]:.1e}")
    ax.set_yscale("log")
    ax.set_xlabel("$t/T$")
    ax.set_ylabel(r"$\kappa\langle|\nabla\theta|^2\rangle$ "
                  r"(normalised by $t=0$)")
    ax.legend(fontsize=6)
    ax.set_title("dissipation peaks when the cascade reaches $\\ell_B$",
                 fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


# --------------------------------------------------------------------------
def fig_compression(out):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 5.6))
    runs = [(bd, load(f"L{CMP['L']}_N{CMP['N']}_bd{bd}")) for bd in CMP["bds"]]
    runs = [(bd, d) for bd, d in runs if d is not None]
    tag = rf"$L={CMP['L']}$, $N={CMP['N']}$"

    # (a) error history at each budget
    ax = axes[0, 0]
    for bd, d in runs:
        ax.semilogy(d["t"], d["err_l2"], color=BD_COLOR[bd],
                    label=rf"$\chi={bd}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$L_2$ error of $\theta$")
    ax.legend(fontsize=6.5)
    ax.set_title(f"error saturates, does not drift ({tag})", fontsize=8)

    # (b) evolution against representation, same budgets
    ax = axes[0, 1]
    for bd, d in runs:
        ax.semilogy(d["t"], d["err_l2"], color=BD_COLOR[bd], lw=1.2)
        ax.semilogy(d["t"], d["repr_err_l2"], color=BD_COLOR[bd], lw=0.9,
                    ls="--")
    ax.plot([], [], "k-", lw=1.2, label="evolution")
    ax.plot([], [], "k--", lw=0.9, label="representation")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$L_2$ error")
    ax.legend(fontsize=6.5)
    ax.set_title("a bounded factor apart throughout", fontsize=8)

    # (c) rank: intrinsic at three probe accuracies, and what the solver carries
    ax = axes[0, 2]
    d = runs[-1][1]
    bound = int(np.max(d["bond_bound"]))
    for t_ in ("1e-06", "0.0001", "0.01"):
        ax.plot(d["t"], intrinsic_chi(d, t_), color=EPS_COLOR[t_],
                label=EPS_LABEL[t_])
    for bd, dd in runs:
        ax.plot(dd["t"], carried_chi(dd), color=BD_COLOR[bd], ls=":", lw=0.9)
    ax.axhline(bound, color="k", lw=0.8, ls="--")
    ax.text(0.02, bound * 0.80, "geometric bound", fontsize=6,
            transform=ax.get_yaxis_transform())
    ax.set_yscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\max_i \chi_i$")
    ax.legend(fontsize=6.5, loc="lower right",
              title="intrinsic (exact field)", title_fontsize=6)
    ax.set_title("rank plateaus; dotted = carried by the solver", fontsize=8)

    # (d) stored parameters
    ax = axes[1, 0]
    for bd, d in runs:
        ax.plot(d["t"], d["tt_params"] / 4.0 ** int(d["L"]), color=BD_COLOR[bd],
                label=rf"$\chi={bd}$")
    for t_ in ("1e-06", "0.01"):
        ax.plot(d["t"], d[f"intrinsic_params_{t_}"] / 4.0 ** int(d["L"]),
                color=EPS_COLOR[t_], ls="--", lw=0.9, label=EPS_LABEL[t_])
    ax.set_yscale("log")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$\rho$")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("cost: carried (solid) vs needed (dashed)", fontsize=8)

    # (e) variance error
    ax = axes[1, 1]
    for bd, d in runs:
        e = np.abs(d["tt_variance"] - d["ref_variance"]) / d["ref_variance"]
        ax.semilogy(d["t"], np.maximum(e, 1e-16), color=BD_COLOR[bd],
                    label=rf"$\chi={bd}$")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$|\langle\theta^2\rangle_{\rm TT}-"
                  r"\langle\theta^2\rangle|/\langle\theta^2\rangle$")
    ax.set_title("variance tracks even when rank-starved", fontsize=8)

    # (f) mass drift: truncation is not mass-conserving
    ax = axes[1, 2]
    for bd, d in runs:
        ax.semilogy(d["t"], np.maximum(np.abs(d["tt_mass"]), 1e-18),
                    color=BD_COLOR[bd], label=rf"$\chi={bd}$")
    ax.semilogy(d["t"], np.maximum(np.abs(d["ref_mass"]), 1e-18), "k-", lw=1.0,
                label="dense")
    ax.set_xlabel("$t$")
    ax.set_ylabel(r"$|\langle\theta\rangle|$")
    ax.legend(fontsize=6, ncol=2)
    ax.set_title("mass drift set by the budget (no projection applied)",
                 fontsize=8)

    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return out


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    print(fig_physical(os.path.join(FIGS, "history_physical.png")))
    print(fig_compression(os.path.join(FIGS, "history_compression.png")))
