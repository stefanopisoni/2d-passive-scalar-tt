import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import pandas as pd

import config as cfg
# This figure only reads precompressed hybrid fields (no spectral_tools / tensnet);
# the raw/compressed data is needed only when PLOT_ONLY=False.

# ── Publication style (match the REVTeX manuscript fonts) ─────────────────────
# This figure spans the full text width of a SINGLE-COLUMN (one-column) article
# (\textwidth = 6.5 in with geometry margin=1in on letterpaper), with the three
# panels laid out side by side. Authored at the true display width with
# Computer-Modern serif fonts and journal-sized text so labels/legends/titles
# match the body text. Uses LaTeX rendering when latex+dvipng are available, and
# falls back to matplotlib's Computer-Modern mathtext otherwise.
import shutil as _shutil
_USE_TEX = _shutil.which('latex') is not None and _shutil.which('dvipng') is not None
mpl.rcParams.update({
    'text.usetex': _USE_TEX,
    'font.family': 'serif',
    'font.serif': ['Computer Modern Roman', 'CMU Serif', 'DejaVu Serif'],
    'mathtext.fontset': 'cm',
    'axes.formatter.use_mathtext': True,
    'font.size': 9,
    'axes.titlesize': 9,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 7,
    'figure.titlesize': 10,
    'lines.linewidth': 1.0,
    'lines.markersize': 3.5,
    'axes.linewidth': 0.6,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'grid.linewidth': 0.4,
})
if _USE_TEX:
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# ── Parameters ────────────────────────────────────────────────────────────────
# PLOT_ONLY=True skips the heavy averaging loop and re-plots from the saved
# figure5_data.pt in `stats_dir`. Set to False to recompute and overwrite it.
PLOT_ONLY = True
L = 12
path = cfg.DATA_DIR          # raw DNS snapshots (only needed if PLOT_ONLY=False)
hyb_dir = cfg.HYB_DIR        # precompressed hybrid TT fields
std_dir = cfg.STD_DIR        # precompressed standard TT fields
stats_dir = cfg.STATS_DIR
downsample = 1
CHI_LIST = [75, 100, 200]               # one panel per bond dimension
N_LIST = [8, 12, 16]                    # hybrid dimension: p1 = 2^N
p_list = [2, 4]                          # need p=2, p=4 for flatness S4/S2^2
r_list = [2**i for i in range(12)]      # x-axis: scales (grid spacing)
configs_to_average = list(range(59, 69))
num_snaps = len(configs_to_average)

# Remaining bond dimension of each hybrid TT (p=4), per (chi, N).
# Taken from hybrid_mps_data.py (hybrid p=4 part): BD_REMAINING values.
BD_REMAINING = {
    75:  {8: 76,  12: 31,  16: 2},
    100: {8: 101, 12: 49,  16: 3},
    200: {8: 201, 12: 138, 16: 11},
}

# ── Styles ──────────────────────────────────────────────────────────────────────
dns_style = ('DNS', 'black', '-', 'o')
# Distinct colours for the hybrid dimension N (legend shows p1 = 2^N).
N_styles = {
    8:  ('orange',   '-.', 'v'),
    12: ('red',      '-.', 'D'),
    16: ('darkred',  '-.', 's'),
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def read_config(config):
    filepath = os.path.join(path, f'theta.{config:05d}')
    theta32 = np.fromfile(filepath, dtype=np.float32)
    theta32 = theta32[1:-1].reshape(4096, -1)
    theta = theta32.view(np.float64)
    return theta

def compute_increments(field, axis=1, r=1, stride=1):
    idx_start = torch.arange(0, field.shape[axis] - r, stride, device=field.device)
    idx_end = idx_start + r
    inc = torch.index_select(field, axis, idx_end) - torch.index_select(field, axis, idx_start)
    return inc

def compute_structure_functions(field_list, r_list, p_list=[2, 4], axis=1, stride=1):
    S = {p: [] for p in p_list}
    for r in r_list:
        all_increments = []
        for field in field_list:
            inc = compute_increments(field, axis=axis, r=r, stride=stride).ravel()
            all_increments.append(inc)
        all_increments = torch.cat(all_increments)
        for p in p_list:
            S[p].append(torch.mean(torch.abs(all_increments)**p).item())
    return {p: torch.tensor(vals) for p, vals in S.items()}

if PLOT_ONLY:
    # ── Load previously computed statistics (skip the heavy averaging loop) ─────
    data_path = os.path.join(stats_dir, 'figure5_data.pt')
    print(f"PLOT_ONLY=True: loading {data_path} ...")
    _d = torch.load(data_path, weights_only=False)
    CHI_LIST    = _d['CHI_LIST']
    N_LIST      = _d['N_LIST']
    r_list      = _d['r_list']
    flatness    = _d['flatness']      # flatness[chi]['dns' or N] -> tensor over r_list
    avg_cr      = _d['avg_cr']        # avg_cr[chi] -> standard p=4 compression ratio
    print(f"Loaded figure5_data.pt (CHI_LIST={CHI_LIST}, N_LIST={N_LIST}).")
else:
    # ── Accumulators (per bond dimension) ───────────────────────────────────────
    # sf_acc[chi]['dns' or N][p] is a tensor over r_list, averaged over snapshots.
    keys = ['dns'] + N_LIST
    sf_acc = {chi: {key: {p: torch.zeros(len(r_list)) for p in p_list} for key in keys}
              for chi in CHI_LIST}
    cr_acc = {chi: 0.0 for chi in CHI_LIST}

    # ── Main averaging loop ──────────────────────────────────────────────────────
    # DNS is bond-dimension independent, so compute it once per snapshot and reuse.
    print(f"Computing flatness vs r for N={N_LIST}, chi={CHI_LIST} over {num_snaps} snaps...")
    for config in configs_to_average:
        snap_str = f"{config:05d}"
        print(f"  Processing snapshot {snap_str}...")

        f_orig = torch.from_numpy(read_config(config))
        sf_dns = compute_structure_functions([f_orig], r_list, p_list=p_list, axis=1, stride=downsample)

        for chi in CHI_LIST:
            for p in p_list:
                sf_acc[chi]['dns'][p] += sf_dns[p]
            # Equivalent compression ratio from the standard MPS p=4 compression at this chi.
            cr_acc[chi] += torch.load(os.path.join(
                std_dir, f'theta{snap_str}_standard_mps_bd{chi}_p4.compression'))
            for N in N_LIST:
                f_hyb = torch.load(os.path.join(
                    hyb_dir, f'theta{snap_str}_hybrid_mps_p4_N{N}_bd_equiv{chi}_int64.tens'))
                sf = compute_structure_functions([f_hyb], r_list, p_list=p_list, axis=1, stride=downsample)
                for p in p_list:
                    sf_acc[chi][N][p] += sf[p]

    # ── Finalize: averages and flatness ──────────────────────────────────────────
    # flatness[chi][key] is a tensor over r_list: F(r) = S4(r) / S2(r)^2
    flatness = {chi: {} for chi in CHI_LIST}
    avg_cr   = {}
    for chi in CHI_LIST:
        for key in keys:
            for p in p_list:
                sf_acc[chi][key][p] /= num_snaps
            flatness[chi][key] = sf_acc[chi][key][4] / (sf_acc[chi][key][2] ** 2)
        avg_cr[chi] = cr_acc[chi] / num_snaps
        print(f"chi={chi}: avg compression ratio (std p=4) = {avg_cr[chi]:.4f}")

    # ── Save statistics ──────────────────────────────────────────────────────────
    os.makedirs(stats_dir, exist_ok=True)
    torch.save({
        'CHI_LIST': CHI_LIST, 'N_LIST': N_LIST, 'r_list': r_list,
        'flatness': flatness, 'avg_cr': avg_cr,
    }, os.path.join(stats_dir, 'figure5_data.pt'))
    print(f"Saved figure5_data.pt in {stats_dir}")

    # CSV: one row per r, columns per (chi, key)
    df = pd.DataFrame({'r': r_list})
    for chi in CHI_LIST:
        df[f'chi{chi}_dns'] = flatness[chi]['dns'].cpu().numpy()
        for N in N_LIST:
            df[f'chi{chi}_N{N}'] = flatness[chi][N].cpu().numpy()
    df.to_csv(os.path.join(stats_dir, 'flatness_vs_N.csv'), index=False)
    print("Saved flatness_vs_N.csv")

# ── Plotting (one row of panels, one per chi) ─────────────────────────────────
fig_dir = cfg.FIG_DIR
os.makedirs(fig_dir, exist_ok=True)

print("Generating figure5...")
# ── Layout (hand-tunable) ─────────────────────────────────────────────────────
# Three panels side by side across the full text width of a one-column article.
FIG_WIDTH    = 6.5    # inches: \textwidth of the single-column layout
FIG_HEIGHT   = 2.6    # inches: panel height (all three panels share it)
PANEL_WPAD  = 1.2     # horizontal gap between panels, in font-size units
                      # (applied by tight_layout, which overrides gridspec wspace)
# Relative physical widths of the panels: a larger value makes that panel wider.
panel_width_ratios = [1.0, 1.0, 1.0]

fig, axes = plt.subplots(1, len(CHI_LIST), figsize=(FIG_WIDTH, FIG_HEIGHT),
                         dpi=300, sharex=True,
                         gridspec_kw={'width_ratios': panel_width_ratios})

# Per-element keyword args (sizes/weights now come from rcParams above)
label_kw  = dict()
# Panel label at 8 pt: the two-line rho <-> chi label is wider than a panel at 9 pt.
panel_kw  = dict(fontsize=8)
# Compact legend so the box stays clear of the curves in the narrow panels.
legend_kw = dict(loc='upper right', handlelength=1.6, labelspacing=0.3,
                 borderpad=0.3, handletextpad=0.5)

r_arr = np.array(r_list, dtype=float)
panel_letters = ['a', 'b', 'c', 'd', 'e', 'f']

# Side-by-side panels (one per bond dimension chi); each keeps its own y-scale.
for j, chi in enumerate(CHI_LIST):
    ax = axes[j]

    # DNS reference
    lbl, col, ls, mk = dns_style
    ax.plot(r_arr, flatness[chi]['dns'].cpu().numpy(), color=col, linestyle=ls,
            marker=mk, markersize=3.5, linewidth=1.0, alpha=0.9, label=lbl, zorder=3)

    # Hybrid methods at different N (legend shows the remaining bond dimension chi).
    for N in N_LIST:
        col, ls, mk = N_styles[N]
        bd_rem = BD_REMAINING[chi][N]
        ax.plot(r_arr, flatness[chi][N].cpu().numpy(), color=col, linestyle=ls,
                marker=mk, markersize=3.5, linewidth=1.0, alpha=0.9,
                label=rf'$p_1 = 2^{{{N}}}$ ($\tilde{{\chi}}={bd_rem}$)', zorder=3)

    ax.set_xscale('log', base=2)
    ax.minorticks_off()
    # Prune top/bottom y-ticks so labels don't collide at the shared panel edges.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    # Panel label on top of the legend: equivalent CR <-> standard-TT bond dimension.
    # Split over two lines: the one-liner is wider than a panel at this width.
    # (x, y) in axes fraction, hand-tunable per panel via `label_pos`.
    label_pos = {0: (0.975, 0.97), 1: (0.975, 0.97), 2: (0.975, 0.97)}
    lx, ly = label_pos[j]
    ax.text(lx, ly,
            rf'{panel_letters[j]}) $\rho={avg_cr[chi]*100:.2f}\%$' '\n'
            rf'$\leftrightarrow \chi_{{\mathrm{{standard\ TT}}}}={chi}$',
            transform=ax.transAxes, ha='right', va='top', linespacing=1.3, **panel_kw)
    ax.grid(True, which='both', alpha=0.3)
    # Legend in every panel (remaining chi is chi-specific), placed below the label.
    legend_pos = {0: (0.975, 0.78), 1: (0.975, 0.78), 2: (0.975, 0.78)}
    ax.legend(bbox_to_anchor=legend_pos[j], **legend_kw)

# Every panel sits on the bottom row now, so each one carries the x label.
for ax in axes:
    ax.set_xlabel(r'$r$', **label_kw)

fig.tight_layout(w_pad=PANEL_WPAD)

fig_png = os.path.join(fig_dir, 'figure5.png')
fig_pdf = os.path.join(fig_dir, 'figure5.pdf')
plt.savefig(fig_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_pdf, bbox_inches='tight')
plt.close()

print(f"Saved figure5.png and figure5.pdf in {fig_dir}")
print("Finished execution.")
