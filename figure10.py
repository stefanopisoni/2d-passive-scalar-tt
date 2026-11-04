"""Figure 10: higher-order normalized structure functions vs compression ratio.

Same layout logic as figure4.py (x-axis: compression ratio rho of each bond
dimension, one column per scale r = 2, 4, 8) but for the three quantities of
figure7.py, one row each: S_4/S_2^2, S_6/S_2^3, S_8/S_2^4. Same snapshots, bond
dimensions, fields and structure-function estimator as figure4.py (horizontal
increments, S_p(r) = <|delta_r theta|^p> averaged over snapshots), so the top row
reproduces Fig. 4.
"""
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator, LogLocator
import pandas as pd

import config as cfg
# spectral_tools (and the raw/compressed data) are only needed to RECOMPUTE the
# cached statistics; imported lazily inside the PLOT_ONLY=False branch below.

# ── Publication style (match the REVTeX manuscript fonts) ─────────────────────
# Full text width of the single-column article (\textwidth = 6.5 in), as Fig. 4.
# Uses LaTeX rendering when latex+dvipng are available, and falls back to
# matplotlib's Computer-Modern mathtext otherwise.
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
# figure10_data.pt in `stats_dir`. Set to False to recompute and overwrite it.
PLOT_ONLY = True
L = 12
path = cfg.DATA_DIR          # raw DNS snapshots (only needed if PLOT_ONLY=False)
std_dir = cfg.STD_DIR        # precompressed standard TT fields
hyb_dir = cfg.HYB_DIR        # precompressed hybrid TT fields
stats_dir = cfg.STATS_DIR
downsample = 1
BD_LIST = [50, 75, 100, 200, 500]    # x-axis: bond dimension (equivalent #params), as Fig. 4
p_list = [2, 4, 6, 8]                 # S_2 for the normalization, S_4/S_6/S_8 for the ratios
p_ratio = [4, 6, 8]                   # one row each: S_p / S_2^(p/2)
r_flatness = [2, 4, 8]               # one column per scale
# configs_to_average = list(range(59, 109)) + list(range(509, 559))
configs_to_average = list(range(59, 69))   # same set as Fig. 4
num_snaps = len(configs_to_average)

# ── Signal catalogue ──────────────────────────────────────────────────────────
# Define label, color, line style, and marker for each model.
signal_styles = {
    'original':            ('DNS',                     'black',      '-',   'o'),
    'galerkin':            ('Galerkin',                'green',      ':',   'v'),
    'wavelet_db32':        ('Wavelet (db32)',          'purple',     '--',  'X'),
    'standard_p4':         ('TT standard (p=4)',       'steelblue',  '--',  's'),
    'hybrid_N12_p4_int64': ('TT hybrid (p=4, N=12, int64)', 'red',    '-.',  'D'),
}
signal_keys = list(signal_styles.keys())
methods = [k for k in signal_keys if k != 'original']   # the compression methods

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
    data_path = os.path.join(stats_dir, 'figure10_data.pt')
    print(f"PLOT_ONLY=True: loading {data_path} ...")
    _d = torch.load(data_path, weights_only=False)
    BD_LIST    = _d['BD_LIST']
    r_flatness = _d['r_flatness']
    sf_ratio   = _d['sf_ratio']
    avg_cr     = _d['avg_cr']
    print(f"Loaded figure10_data.pt (BD_LIST={BD_LIST}).")
else:
    # Recompute path: needs spectral_tools + the raw/compressed data on disk.
    from spectral_tools import galerkin_truncate_2d, wavelet_truncate_2d_pywt
    # ── Accumulators (per bond dimension) ───────────────────────────────────────
    # sf_acc[bd][k][p] is a tensor over r_flatness, averaged over snapshots.
    sf_acc = {bd: {k: {p: torch.zeros(len(r_flatness)) for p in p_list} for k in signal_keys} for bd in BD_LIST}
    cr_acc = {bd: 0.0 for bd in BD_LIST}

    # ── Main averaging loop ──────────────────────────────────────────────────────
    for bd in BD_LIST:
        print(f"=== Bond dimension BD={bd} ===")
        for config in configs_to_average:
            snap_str = f"{config:05d}"
            print(f"  Processing snapshot {snap_str}...")

            # Load original
            theta_raw = read_config(config)
            f_orig = torch.from_numpy(theta_raw)

            # DOF target from the standard p=4 compression ratio (matches all methods)
            std_cr = torch.load(os.path.join(std_dir, f'theta{snap_str}_standard_mps_bd{bd}_p4.compression'))
            cr_acc[bd] += std_cr
            dof_target = std_cr * (2**24)

            # DOF-matched spectral / wavelet truncations
            f_galerkin, _, _ = galerkin_truncate_2d(f_orig, dof_target)
            f_wavelet_db32   = wavelet_truncate_2d_pywt(f_orig, dof_target, wavelet='db32')

            # Pre-compressed standard and hybrid MPS fields
            f_std_p4       = torch.load(os.path.join(std_dir, f'theta{snap_str}_standard_mps_bd{bd}_p4.tens'))
            f_hybrid_int64 = torch.load(os.path.join(hyb_dir, f'theta{snap_str}_hybrid_mps_p4_N12_bd_equiv{bd}_int64.tens'))

            fields = {
                'original':            f_orig,
                'galerkin':            f_galerkin,
                'wavelet_db32':        f_wavelet_db32,
                'standard_p4':         f_std_p4,
                'hybrid_N12_p4_int64': f_hybrid_int64,
            }

            for k, f in fields.items():
                sf = compute_structure_functions([f], r_flatness, p_list=p_list, axis=1, stride=downsample)
                for p in p_list:
                    sf_acc[bd][k][p] += sf[p]

    # ── Finalize: averages and normalized structure functions ────────────────────
    # sf_ratio[bd][k][p] is a tensor over r_flatness: S_p(r) / S_2(r)^(p/2)
    # (p=4 is the flatness of Fig. 4)
    sf_ratio = {bd: {} for bd in BD_LIST}
    avg_cr   = {}
    for bd in BD_LIST:
        for k in signal_keys:
            for p in p_list:
                sf_acc[bd][k][p] /= num_snaps
            sf_ratio[bd][k] = {p: sf_acc[bd][k][p] / (sf_acc[bd][k][2] ** (p // 2)) for p in p_ratio}
        avg_cr[bd] = cr_acc[bd] / num_snaps
        print(f"BD={bd}: avg compression ratio (std p=4) = {avg_cr[bd]:.4f}")

    # ── Save statistics ──────────────────────────────────────────────────────────
    os.makedirs(stats_dir, exist_ok=True)
    torch.save({
        'BD_LIST': BD_LIST, 'r_flatness': r_flatness, 'configs': configs_to_average,
        'sf_acc': sf_acc, 'sf_ratio': sf_ratio, 'avg_cr': avg_cr,
    }, os.path.join(stats_dir, 'figure10_data.pt'))
    print(f"Saved figure10_data.pt in {stats_dir}")

    # CSV: one row per BD, columns per method/order/scale
    rows = []
    for bd in BD_LIST:
        row = {'BD': bd, 'avg_cr': float(avg_cr[bd])}
        for k in signal_keys:
            for p in p_ratio:
                for j, r in enumerate(r_flatness):
                    row[f'{k}_S{p}_over_S2^{p // 2}_r{r}'] = float(sf_ratio[bd][k][p][j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(stats_dir, 'structure_functions_high_order_vs_bd.csv'), index=False)
    print("Saved structure_functions_high_order_vs_bd.csv")

# ── Plotting (3x3: rows = S_p/S_2^(p/2), columns = scale r) ───────────────────
fig_dir = cfg.FIG_DIR
os.makedirs(fig_dir, exist_ok=True)

print("Generating figure10...")
label_kw  = dict()
panel_kw  = dict(fontsize=9)

# x-axis: compression ratio rho (%) corresponding to each bond dimension.
rho_arr = np.array([avg_cr[bd] * 100 for bd in BD_LIST])
panel_letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']
row_ylabels = {4: r'$S_4/S_2^2$', 6: r'$S_6/S_2^3$', 8: r'$S_8/S_2^4$'}

y_dns    = {(i, j): float(sf_ratio[BD_LIST[0]]['original'][p][j])
            for i, p in enumerate(p_ratio) for j in range(len(r_flatness))}
y_curves = {(i, j): {k: np.array([float(sf_ratio[bd][k][p][j]) for bd in BD_LIST]) for k in methods}
            for i, p in enumerate(p_ratio) for j in range(len(r_flatness))}

# Per-row y scale: the higher orders can span decades across rho, so they may need
# a log axis. Per-panel y-limits: each panel brackets its OWN data range (as Fig. 4),
# padded by `y_margin_frac` (in log space on log rows); override an entry in
# `y_lims_override` ((row, col) -> (lo, hi)) to set a panel's limits by hand.
row_yscale       = ['linear', 'log', 'log']        # one per row: S4/S2^2, S6/S2^3, S8/S2^4
y_margin_frac    = 0.10
y_lims_override  = {}                    # e.g. {(2, 0): (10, 1e4)}
y_lims = {}
for (i, j), curves in y_curves.items():
    if (i, j) in y_lims_override:
        y_lims[(i, j)] = y_lims_override[(i, j)]
        continue
    vals = np.concatenate(list(curves.values()) + [[y_dns[(i, j)]]])
    if row_yscale[i] == 'log':
        lo, hi = np.log10(vals.min()), np.log10(vals.max())
        m = y_margin_frac * (hi - lo)
        y_lims[(i, j)] = (10**(lo - m), 10**(hi + m))
    else:
        m = y_margin_frac * (vals.max() - vals.min())
        y_lims[(i, j)] = (vals.min() - m, vals.max() + m)

# ── Layout (hand-tunable) ─────────────────────────────────────────────────────
FIG_WIDTH   = 6.5     # inches: \textwidth of the single-column layout
FIG_HEIGHT  = 6.0     # inches: three rows of Fig. 4-sized panels
PANEL_WPAD  = 1.2     # horizontal gap between panels, in font-size units
PANEL_HPAD  = 0.6     # vertical gap between rows, in font-size units
XTICK_ROTATION = 45   # rho tick labels are tight in a narrow panel; 0 = horizontal
# Panel label position (axes fraction, ha/va='center'), (row, col) -> (x, y).
# Panel a) holds the legend in its top-right, so its label is dropped lower.
label_pos = {(i, j): (0.72, 0.90) for i in range(len(p_ratio)) for j in range(len(r_flatness))}
label_pos[(0, 0)] = (0.72, 0.47)

fig, axes = plt.subplots(len(p_ratio), len(r_flatness),
                         figsize=(FIG_WIDTH, FIG_HEIGHT),
                         dpi=300, sharex=True)

for i, p in enumerate(p_ratio):
    for j, r in enumerate(r_flatness):
        ax = axes[i, j]

        # DNS reference: bond-dimension independent -> horizontal line.
        ax.axhline(y_dns[(i, j)], color='black', linestyle='--', linewidth=1.0, alpha=0.6,
                   zorder=1, label='DNS (reference)')

        for k in methods:
            lbl, color, ls, mk = signal_styles[k]
            ax.plot(rho_arr, y_curves[(i, j)][k], color=color, linestyle=ls, marker=mk,
                    markersize=3.5, linewidth=1.0, alpha=0.9, label=lbl, zorder=3)

        ax.set_xscale('log')
        ax.set_yscale(row_yscale[i])
        ax.set_xticks(rho_arr)
        ax.set_xticklabels([f'{v:.2f}' for v in rho_arr], rotation=XTICK_ROTATION,
                           ha='right' if XTICK_ROTATION else 'center',
                           rotation_mode='anchor' if XTICK_ROTATION else None)
        ax.minorticks_off()
        ax.set_ylim(*y_lims[(i, j)])
        if row_yscale[i] == 'linear':
            # Prune top/bottom y-ticks so labels don't collide at the shared panel edges.
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
        else:
            # Keep the y minor ticks on log rows (minorticks_off above is for the
            # rho axis): panels spanning < 1 decade get labelled minor ticks.
            ax.yaxis.set_minor_locator(LogLocator(base=10, subs='auto'))
        lx, ly = label_pos[(i, j)]
        ax.text(lx, ly, f'{panel_letters[i * len(r_flatness) + j]}) $r={r}$',
                transform=ax.transAxes, ha='center', va='center', **panel_kw)
        ax.grid(True, which='both', alpha=0.3)

    axes[i, 0].set_ylabel(row_ylabels[p], **label_kw)

for ax in axes[-1]:
    ax.set_xlabel(r'$\rho\ (\%)$', **label_kw)

# Legend inside panel a), as in Fig. 4. Parenthetical detail is dropped from the
# labels (given in the caption) except the wavelet basis (db32), which is kept.
handles, labels = axes[0, 0].get_legend_handles_labels()
labels = [l if l.startswith('Wavelet') else re.sub(r'\s*\(.*\)', '', l) for l in labels]
axes[0, 0].legend(handles, labels, loc='upper right', handlelength=1.8,
                  labelspacing=0.3, borderpad=0.3)
fig.tight_layout(w_pad=PANEL_WPAD, h_pad=PANEL_HPAD)

fig_png = os.path.join(fig_dir, 'figure10.png')
fig_pdf = os.path.join(fig_dir, 'figure10.pdf')
plt.savefig(fig_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_pdf, bbox_inches='tight')
plt.close()

print(f"Saved figure10.png and figure10.pdf in {fig_dir}")
print("Finished execution.")
