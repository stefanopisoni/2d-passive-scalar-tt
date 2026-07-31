import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, MaxNLocator
import pandas as pd

import config as cfg
# spectral_tools (and the raw/compressed data) are only needed to RECOMPUTE the
# cached statistics; imported lazily inside the PLOT_ONLY=False branch below.

# ── Publication style (match the REVTeX/PRL manuscript fonts) ──────────────────
# This is a single-column figure (\columnwidth = 3.125 in with the manuscript's
# revtex4-2 prl + geometry margin=1in). Authored at the true display width with
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
# figure4_data.pt in `stats_dir`. Set to False to recompute and overwrite it.
PLOT_ONLY = True
L = 12
path = cfg.DATA_DIR          # raw DNS snapshots (only needed if PLOT_ONLY=False)
std_dir = cfg.STD_DIR        # precompressed standard TT fields
hyb_dir = cfg.HYB_DIR        # precompressed hybrid TT fields
stats_dir = cfg.STATS_DIR
downsample = 1
BD_LIST = [50, 75, 100, 200, 500]    # x-axis: bond dimension (equivalent #params)
p_list = [2, 4]                       # need p=2, p=4 for flatness S4/S2^2
r_flatness = [2, 4, 8]               # one curve per method per scale
bins = torch.arange(-20, 20, 0.1)
# configs_to_average = list(range(59, 109)) + list(range(509, 559))
configs_to_average = list(range(59, 69))
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

# Per-scale line style / marker (color encodes the method instead)
r_styles = {2: ('-', 'o'), 4: ('--', 's'), 8: (':', '^')}

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
    data_path = os.path.join(stats_dir, 'figure4_data.pt')
    print(f"PLOT_ONLY=True: loading {data_path} ...")
    _d = torch.load(data_path, weights_only=False)
    BD_LIST    = _d['BD_LIST']
    r_flatness = _d['r_flatness']
    flatness   = _d['flatness']
    avg_cr     = _d['avg_cr']
    print(f"Loaded figure4_data.pt (BD_LIST={BD_LIST}).")
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

    # ── Finalize: averages and flatness ──────────────────────────────────────────
    # flatness[bd][k] is a tensor over r_flatness: F(r) = S4(r) / S2(r)^2
    flatness = {bd: {} for bd in BD_LIST}
    avg_cr   = {}
    for bd in BD_LIST:
        for k in signal_keys:
            for p in p_list:
                sf_acc[bd][k][p] /= num_snaps
            flatness[bd][k] = sf_acc[bd][k][4] / (sf_acc[bd][k][2] ** 2)
        avg_cr[bd] = cr_acc[bd] / num_snaps
        print(f"BD={bd}: avg compression ratio (std p=4) = {avg_cr[bd]:.4f}")

    # ── Save statistics ──────────────────────────────────────────────────────────
    os.makedirs(stats_dir, exist_ok=True)
    torch.save({
        'BD_LIST': BD_LIST, 'r_flatness': r_flatness,
        'flatness': flatness, 'avg_cr': avg_cr,
    }, os.path.join(stats_dir, 'figure4_data.pt'))
    print(f"Saved figure4_data.pt in {stats_dir}")

    # CSV: one row per BD, columns per method/scale
    rows = []
    for bd in BD_LIST:
        row = {'BD': bd, 'avg_cr': float(avg_cr[bd])}
        for k in signal_keys:
            for j, r in enumerate(r_flatness):
                row[f'{k}_r{r}'] = float(flatness[bd][k][j])
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(stats_dir, 'flatness_vs_bd.csv'), index=False)
    print("Saved flatness_vs_bd.csv")

# ── Plotting (single figure: flatness vs bond dimension) ──────────────────────────
fig_dir = cfg.FIG_DIR
os.makedirs(fig_dir, exist_ok=True)

print("Generating figure4...")
label_kw  = dict()
panel_kw  = dict(fontsize=9)
legend_kw = dict(loc='upper right')

# x-axis: compression ratio rho (%) corresponding to each bond dimension.
rho_arr = np.array([avg_cr[bd] * 100 for bd in BD_LIST])
panel_letters = ['a', 'b', 'c', 'd', 'e', 'f']

y_dns   = {j: float(flatness[BD_LIST[0]]['original'][j]) for j in range(len(r_flatness))}
y_curves = {j: {k: np.array([float(flatness[bd][k][j]) for bd in BD_LIST]) for k in methods}
            for j in range(len(r_flatness))}

# Per-panel y-limits: each panel brackets its OWN data range, so the vertical scale
# is INDEPENDENT per panel. This spreads out the r=8 curves (which sit close
# together) instead of forcing them onto the r=2 panel's units-per-inch -> the same
# delta in flatness spans a different number of pixels in each panel.
# `y_margin_frac` pads above/below the data; override an entry in `y_lims_override`
# (j -> (lo, hi)) to set a panel's limits by hand.
y_margin_frac    = 0.10
y_lims_override  = {}                    # e.g. {2: (8, 21)} to pin the r=8 panel
y_lims = {}
for j in range(len(r_flatness)):
    if j in y_lims_override:
        y_lims[j] = y_lims_override[j]
        continue
    vals = np.concatenate(list(y_curves[j].values()) + [[y_dns[j]]])
    m = y_margin_frac * (vals.max() - vals.min())
    y_lims[j] = (vals.min() - m, vals.max() + m)

# Relative physical heights of the stacked panels (hand-tunable): a larger value
# makes that panel taller, spreading its curves further apart.
panel_height_ratios = [1.5, 1.0, 1.2]

fig, axes = plt.subplots(len(r_flatness), 1,
                         figsize=(3.125, 1.15 * sum(panel_height_ratios) + 0.45),
                         dpi=300, sharex=True,
                         gridspec_kw={'hspace': 0, 'height_ratios': panel_height_ratios})

# Stacked, contiguous panels (one per scale r); single shared x-axis at the bottom.
for j, r in enumerate(r_flatness):
    ax = axes[j]

    # DNS reference: bond-dimension independent -> horizontal line.
    ax.axhline(y_dns[j], color='black', linestyle='--', linewidth=1.0, alpha=0.6,
               zorder=1, label='DNS (reference)')

    for k in methods:
        lbl, color, ls, mk = signal_styles[k]
        ax.plot(rho_arr, y_curves[j][k], color=color, linestyle=ls, marker=mk,
                markersize=3.5, linewidth=1.0, alpha=0.9, label=lbl, zorder=3)

    ax.set_xscale('log')
    ax.set_xticks(rho_arr)
    ax.set_xticklabels([f'{v:.2f}' for v in rho_arr])
    ax.minorticks_off()
    ax.set_ylim(*y_lims[j])
    # Prune top/bottom y-ticks so labels don't collide at the shared panel edges.
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4, prune='both'))
    # Panel label position (axes fraction, ha/va='center'), independent per panel.
    # Tune each entry by hand: j=0 -> a) r=2, j=1 -> b) r=4, j=2 -> c) r=8.
    # Panel a's legend occupies the top-right, so its label is dropped lower.
    label_pos = {0: (0.85, 0.46), 1: (0.85, 0.85), 2: (0.85, 0.75)}
    lx, ly = label_pos[j]
    ax.text(lx, ly, f'{panel_letters[j]}) $r={r}$', transform=ax.transAxes,
            ha='center', va='center', **panel_kw)
    ax.grid(True, which='both', alpha=0.3)

axes[-1].set_xlabel(r'$\rho\ (\%)$', **label_kw)

# Legend inside the top (r=2) panel, top-right corner where the curves have
# converged low. Parenthetical detail is dropped from the labels (given in the
# caption) except the wavelet basis (db32), which is kept.
import re
handles, labels = axes[0].get_legend_handles_labels()
labels = [l if l.startswith('Wavelet') else re.sub(r'\s*\(.*\)', '', l) for l in labels]
axes[0].legend(handles, labels, loc='upper right', handlelength=1.8,
               labelspacing=0.3, borderpad=0.3)
fig.tight_layout()

fig_png = os.path.join(fig_dir, 'figure4.png')
fig_pdf = os.path.join(fig_dir, 'figure4.pdf')
plt.savefig(fig_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_pdf, bbox_inches='tight')
plt.close()

print(f"Saved figure4.png and figure4.pdf in {fig_dir}")
print("Finished execution.")
