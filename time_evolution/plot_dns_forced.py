"""Figures for the DNS-velocity ladder and the forced steady-state run.

decay_DNS9_N*_bd*.npz  scalar advected by a frozen DNS velocity snapshot, 512^2
decay_FRC8_N*_bd*.npz  cellular flow with a steady source, 256^2

Writes figures/dns_advection.png and figures/forced_steady.png.
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
N_COLOR = {0: "k", 6: "#2ca02c", 8: "#d62728", 10: "#9467bd", 12: "#8c564b"}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 150, "savefig.bbox": "tight",
                     "legend.frameon": False})


def collect(tag):
    pat = re.compile(rf"decay_{tag}_N(\d+)_bd(\d+)\.npz$")
    out = {}
    for fn in sorted(glob.glob(os.path.join(RESULTS, f"decay_{tag}_*.npz"))):
        m = pat.search(os.path.basename(fn))
        if not m:
            continue
        N, bd = int(m.group(1)), int(m.group(2))
        d = np.load(fn)
        t = d["t"]
        late = t >= (2.0 / 3.0) * t[-1]
        out.setdefault(N, []).append(dict(
            N=N, bd=bd, d=d, t=t, late=late,
            rho=float(np.mean(d["tt_params"][late]) / float(d["n_cells"])),
            err=float(np.mean(d["err_l2"][late])),
            rep=float(np.mean(d["repr_err_l2"][late])),
        ))
    for N in out:
        out[N].sort(key=lambda r: r["bd"])
    return out


def label(N):
    return "standard ($N{=}0$)" if N == 0 else rf"$N{{=}}{N}$"


def fig_dns(data):
    if not data:
        return None
    fig = plt.figure(figsize=(11, 6.4))
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
    any_run = next(iter(data.values()))[0]["d"]
    L = int(any_run["L"])

    ax = fig.add_subplot(gs[0, 0])
    for N in sorted(data):
        r = data[N]
        ax.loglog([x["rho"] for x in r], [x["err"] for x in r], "-o", ms=4, lw=1.4,
                  color=N_COLOR.get(N, "0.5"), label=label(N))
    ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$L_2$ error")
    ax.legend(fontsize=6.5)
    ax.set_title(rf"evolution error, DNS velocity, $L={L}$", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    for N in sorted(data):
        r = data[N]
        ax.loglog([x["rho"] for x in r], [x["rep"] for x in r], "--s", ms=3.5,
                  lw=1.2, color=N_COLOR.get(N, "0.5"), label=label(N))
    ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$L_2$ error")
    ax.legend(fontsize=6.5)
    ax.set_title("representation error at the same budget", fontsize=9)

    # intrinsic rank of the true solution
    ax = fig.add_subplot(gs[0, 2])
    d = any_run
    for e, c in (("1e-06", "#1f77b4"), ("0.0001", "#ff7f0e"), ("0.01", "#2ca02c")):
        key = f"intrinsic_bonds_{e}"
        if key in d.files:
            ax.plot(d["t"], d[key].max(axis=1), lw=1.4, color=c,
                    label=rf"$\epsilon={float(e):g}$")
    ax.axhline(d["bond_bound"].max(), color="0.4", ls="--", lw=1)
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\max_j\chi_j$")
    ax.legend(fontsize=7)
    ax.set_title("intrinsic rank of the true solution", fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    k = np.arange(d["ref_E"].shape[1])
    cm = plt.cm.viridis(np.linspace(0, 0.9, 5))
    idx = np.linspace(0, len(d["t"]) - 1, 5).astype(int)
    for c, i in zip(cm, idx):
        E = d["ref_E"][i]
        m = (k > 0) & (E > 0)
        ax.loglog(k[m], E[m], color=c, lw=1.2, label=rf"$t={d['t'][i]:.2f}$")
    ax.set_xlabel("$k$"); ax.set_ylabel(r"$E_\theta(k)$")
    ax.set_ylim(1e-12, None); ax.legend(fontsize=6, ncol=2)
    ax.set_title("variance spectrum (dense truth)", fontsize=9)

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(d["t"], d["ref_variance"], "k-", lw=1.6, label=r"$\langle\theta^2\rangle$")
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\langle\theta^2\rangle$")
    ax2 = ax.twinx()
    ax2.plot(d["t"], d["ref_chi"], color="#d62728", lw=1.2, label=r"$\chi$")
    ax2.set_ylabel(r"$\chi$", color="#d62728", labelpad=1)
    ax2.tick_params(axis="y", colors="#d62728", labelsize=7)
    ax2.locator_params(axis="y", nbins=4); ax2.grid(False)
    ax.set_title("decay and dissipation", fontsize=9)

    ax = fig.add_subplot(gs[1, 2])
    for N in sorted(data):
        for x in data[N]:
            if x["bd"] != 32:
                continue
            ax.semilogy(x["t"], x["d"]["err_l2"], lw=1.3,
                        color=N_COLOR.get(N, "0.5"), label=label(N))
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$L_2$ error")
    ax.legend(fontsize=6.5); ax.set_title(r"error history, $\chi=32$", fontsize=9)

    fig.suptitle("Scalar advected by a frozen DNS velocity snapshot "
                 r"($512^2$, $\psi$ truncated at $10^{-2}$)", fontsize=10, y=0.98)
    out = os.path.join(FIGS, "dns_advection.png")
    fig.savefig(out); plt.close(fig)
    return out


def fig_forced(data):
    if not data:
        return None
    fig = plt.figure(figsize=(11, 3.4))
    gs = fig.add_gridspec(1, 3, wspace=0.32)
    any_run = next(iter(data.values()))[0]["d"]

    ax = fig.add_subplot(gs[0, 0])
    d = any_run
    ax.plot(d["t"], d["ref_variance"], "k-", lw=1.6)
    t0 = 0.6 * d["t"][-1]
    ax.axvspan(t0, d["t"][-1], color="#1f77b4", alpha=0.12)
    ax.text(0.97, 0.08, "window used", fontsize=7, ha="right", color="#1f77b4",
            transform=ax.transAxes)
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\langle\theta^2\rangle$")
    ax.set_title("variance reaches a plateau", fontsize=9)

    ax = fig.add_subplot(gs[0, 1])
    for e, c in (("1e-06", "#1f77b4"), ("0.0001", "#ff7f0e"), ("0.01", "#2ca02c")):
        key = f"intrinsic_bonds_{e}"
        if key in d.files:
            ax.plot(d["t"], d[key].max(axis=1), lw=1.4, color=c,
                    label=rf"$\epsilon={float(e):g}$")
    ax.axhline(d["bond_bound"].max(), color="0.4", ls="--", lw=1)
    ax.set_xlabel("$t$"); ax.set_ylabel(r"$\max_j\chi_j$")
    ax.legend(fontsize=7)
    ax.set_title("intrinsic rank in the steady state", fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    for N in sorted(data):
        r = data[N]
        ax.loglog([x["rho"] for x in r], [x["err"] for x in r], "-o", ms=4, lw=1.4,
                  color=N_COLOR.get(N, "0.5"), label=label(N))
    ax.set_xlabel(r"$\rho$"); ax.set_ylabel(r"$L_2$ error")
    ax.legend(fontsize=6.5); ax.set_title("evolution error vs cost", fontsize=9)

    fig.suptitle(r"Forced cellular flow, statistically steady ($256^2$, "
                 r"$f=0.8\sin 4y$)", fontsize=10, y=1.02)
    out = os.path.join(FIGS, "forced_steady.png")
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def report(tag, data):
    if not data:
        print(f"[{tag}] no runs"); return
    print(f"\n=== {tag} ===")
    print(f"{'N':>3}{'chi':>5}{'rho':>9}{'evo L2':>11}{'repr L2':>11}{'ratio':>7}")
    for N in sorted(data):
        for x in data[N]:
            print(f"{x['N']:>3}{x['bd']:>5}{x['rho']:>9.4f}{x['err']:>11.2e}"
                  f"{x['rep']:>11.2e}{x['err']/max(x['rep'],1e-300):>7.1f}")
    errs = np.logspace(-4, 0, 60)

    def rho_at(rows, e):
        r = sorted(rows, key=lambda x: x["err"])
        ee = np.array([x["err"] for x in r]); pp = np.array([x["rho"] for x in r])
        return np.nan if e < ee[0] or e > ee[-1] else float(
            np.exp(np.interp(np.log(e), np.log(ee), np.log(pp))))

    if 0 in data:
        print("  budget saved at fixed accuracy:")
        for te in (3e-1, 1e-1, 3e-2, 1e-2):
            s = rho_at(data[0], te)
            cand = [(N, rho_at(data[N], te)) for N in data if N > 0]
            cand = [(N, v) for N, v in cand if np.isfinite(v)]
            if np.isfinite(s) and cand:
                N, v = min(cand, key=lambda z: z[1])
                print(f"    err={te:.0e}: {s/v:.1f}x  (N={N})")


if __name__ == "__main__":
    os.makedirs(FIGS, exist_ok=True)
    dns, frc = collect("DNS9"), collect("FRC8")
    for o in (fig_dns(dns), fig_forced(frc)):
        if o:
            print(o)
    report("DNS velocity, L=9", dns)
    report("Forced cellular, L=8", frc)
