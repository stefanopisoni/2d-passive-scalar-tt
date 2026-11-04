"""Figure 7: higher-order normalized structure functions S_4/S_2^2, S_6/S_2^3, S_8/S_2^4.

Same snapshots, fields and structure-function estimator as figure1.py (S_p(r) =
<|delta_r theta|^p>, horizontal increments, averaged over snapshots), extended to
p = 6, 8. The S_4/S_2^2 panel therefore reproduces the flatness of Fig. 1 c).
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import config as cfg
# spectral_tools (and the raw/compressed data) are only needed to RECOMPUTE the
# cached statistics; imported lazily inside the PLOT_ONLY=False branch below.

# ── Publication style (match the REVTeX/PRL manuscript fonts) ──────────────────
# The manuscript is compiled with revtex4-2 (prl) + geometry margin=1in, giving
# \textwidth = 6.5 in and \columnwidth = 3.125 in. Figures are therefore authored
# at their true display width with Computer-Modern serif fonts and journal-sized
# text, so that nothing is shrunk on inclusion and the labels/legends/titles match
# the body text. Uses LaTeX rendering when a latex+dvipng pipeline is available,
# and falls back to matplotlib's Computer-Modern mathtext otherwise.
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
    'xtick.labelsize': 6.5,
    'ytick.labelsize': 6.5,
    'xtick.major.size': 2.5,
    'ytick.major.size': 2.5,
    'xtick.minor.size': 1.5,
    'ytick.minor.size': 1.5,
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
# figure7_data.pt in `stats_dir`. Set to False to recompute and overwrite it.
PLOT_ONLY = True
L = 12
path = cfg.DATA_DIR          # raw DNS snapshots (only needed if PLOT_ONLY=False)
std_dir = cfg.STD_DIR        # precompressed standard TT fields
hyb_dir = cfg.HYB_DIR        # precompressed hybrid TT fields
stats_dir = cfg.STATS_DIR
downsample = 1
BD_MAX = 100
p_list = [2, 4, 6, 8]        # S_2 for the normalization, S_4/S_6/S_8 for the ratios
p_ratio = [4, 6, 8]          # plotted: S_p / S_2^(p/2)
r_list = [2**i for i in range(12)]
r_pdf_marks = [2, 4, 8]   # scales at which figure2.py plots the increment PDFs
n_hybrid_uncompressed = 6  # the hybrid MPS leaves the 6 biggest scales uncompressed
configs_to_average = list(range(59, 109)) + list(range(509, 559))   # same set as Fig. 1
num_snaps = len(configs_to_average)

# ── Signal catalogue ──────────────────────────────────────────────────────────
# Define label, color, line style, and marker for the 5 selected models
signal_styles = {
    'original':            ('DNS',                     'black',      '-',   'o'),
    'galerkin':            ('Galerkin',                'green',      ':',   'v'),
    'wavelet_db32':        ('Wavelet (db32)',          'purple',     '--',  'X'),
    'standard_p4':         ('TT standard',             'steelblue',  '--',  's'),
    'hybrid_N12_p4_int64': ('TT hybrid',               'red',        '-.',  'D'),
}
signal_keys = list(signal_styles.keys())

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
    data_path = os.path.join(stats_dir, 'figure7_data.pt')
    print(f"PLOT_ONLY=True: loading {data_path} ...")
    _d = torch.load(data_path, weights_only=False)
    r_list   = _d['r_list']
    sf_acc   = _d['sf_acc']
    sf_ratio = _d['sf_ratio']
    avg_cr   = _d['avg_cr']
    print(f"Loaded figure7_data.pt (avg CR = {avg_cr:.4f}).")
else:
    # Recompute path: needs spectral_tools + the raw/compressed data on disk.
    from spectral_tools import galerkin_truncate_2d, wavelet_truncate_2d_pywt
    # ── Accumulators ────────────────────────────────────────────────────────────
    sf_acc = {k: {p: torch.zeros(len(r_list)) for p in p_list} for k in signal_keys}
    cr_acc = 0.0

    # ── Main Averaging Loop ───────────────────────────────────────────────────────
    print(f"Starting calculation for BD_MAX={BD_MAX} across {num_snaps} snapshots...")
    for config in configs_to_average:
        snap_str = f"{config:05d}"
        print(f"Processing snapshot {snap_str}...")

        # Load original
        theta_raw = read_config(config)
        f_orig = torch.from_numpy(theta_raw)

        # DOF target from standard p=4 compression ratio (for Galerkin/wavelet matching)
        std_p4_cr_path = os.path.join(std_dir, f'theta{snap_str}_standard_mps_bd{BD_MAX}_p4.compression')
        std_p4_cr = torch.load(std_p4_cr_path)
        cr_acc += std_p4_cr
        dof_target = std_p4_cr * (2**24)

        # Compute Galerkin and Wavelet truncations
        f_galerkin, _, _ = galerkin_truncate_2d(f_orig, dof_target)
        f_wavelet_db32 = wavelet_truncate_2d_pywt(f_orig, dof_target, wavelet='db32')

        # Load pre-compressed standard and hybrid MPS fields
        std_p4_path = os.path.join(std_dir, f'theta{snap_str}_standard_mps_bd{BD_MAX}_p4.tens')
        hybrid_int64_path = os.path.join(hyb_dir, f'theta{snap_str}_hybrid_mps_p4_N12_bd_equiv{BD_MAX}_int64.tens')

        f_std_p4 = torch.load(std_p4_path)
        f_hybrid_int64 = torch.load(hybrid_int64_path)

        fields = {
            'original':            f_orig,
            'galerkin':            f_galerkin,
            'wavelet_db32':        f_wavelet_db32,
            'standard_p4':         f_std_p4,
            'hybrid_N12_p4_int64': f_hybrid_int64,
        }

        # Accumulate structure functions for each signal
        for k, f in fields.items():
            sf = compute_structure_functions([f], r_list, p_list=p_list, axis=1, stride=downsample)
            for p in p_list:
                sf_acc[k][p] += sf[p]

    # ── Finalize Averages ──────────────────────────────────────────────────────────
    avg_cr = cr_acc / num_snaps
    print(f"Finalizing averages. Average compression ratio (std p=4, BD={BD_MAX}): {avg_cr:.4f}")

    for k in signal_keys:
        for p in p_list:
            sf_acc[k][p] /= num_snaps

    # Normalized structure functions S_p(r) / S_2(r)^(p/2)  (p=4 is the flatness of Fig. 1)
    sf_ratio = {k: {p: sf_acc[k][p] / (sf_acc[k][2] ** (p // 2)) for p in p_ratio}
                for k in signal_keys}

    # ── Save Statistics ───────────────────────────────────────────────────────────
    os.makedirs(stats_dir, exist_ok=True)

    # 1. Save consolidated PyTorch file
    pytorch_data = {
        'r_list': r_list,
        'sf_acc': sf_acc,
        'sf_ratio': sf_ratio,
        'avg_cr': avg_cr
    }
    torch.save(pytorch_data, os.path.join(stats_dir, 'figure7_data.pt'))
    print(f"Saved figure7_data.pt in {stats_dir}")

    # 2. Save structure functions and their normalized ratios to CSV
    df_sf = pd.DataFrame({'r': r_list})
    for k in signal_keys:
        for p in p_list:
            df_sf[f'{k}_S{p}'] = sf_acc[k][p].cpu().numpy()
        for p in p_ratio:
            df_sf[f'{k}_S{p}_over_S2^{p // 2}'] = sf_ratio[k][p].cpu().numpy()
    df_sf.to_csv(os.path.join(stats_dir, 'structure_functions_high_order.csv'), index=False)
    print("Saved structure_functions_high_order.csv")

# ── Plotting (1 Row x 3 Columns) ──────────────────────────────────────────────
fig_dir = cfg.FIG_DIR
os.makedirs(fig_dir, exist_ok=True)

print("Generating higher-order structure function figure...")
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4), dpi=300)

# Per-element keyword args (sizes/weights now come from rcParams above)
label_kw = dict()

# Per-panel y scale: a) linear as the flatness panel of Fig. 1; the higher orders
# span several decades, so they are drawn on a log axis.
panel_yscale = ['linear', 'log', 'log']

# Shaded region marking the 6 biggest scales that the TT hybrid leaves uncompressed.
# Left boundary is the smallest uncompressed scale; drawn in the TT hybrid colour,
# almost transparent.
hybrid_color = signal_styles['hybrid_N12_p4_int64'][1]
r_sorted = sorted(r_list)
hybrid_shade_left = r_sorted[-n_hybrid_uncompressed]

def shade_hybrid_uncompressed(ax):
    """Shade the right part of the r-axis (the 6 biggest, uncompressed scales)."""
    xlim = ax.get_xlim()
    ax.axvspan(hybrid_shade_left, xlim[1], color=hybrid_color, alpha=0.08, zorder=0,
               label='TT hybrid - exact')
    ax.set_xlim(xlim)

for ax, p, yscale in zip(axes, p_ratio, panel_yscale):
    for i, r in enumerate(r_pdf_marks):
        ax.axvline(r, color='grey', linestyle='-', linewidth=2.5, alpha=0.25, zorder=0,
                   label='PDF scales (Fig. 2)' if i == 0 else None)
    for k in signal_keys:
        lbl, col, ls, mk = signal_styles[k]
        ax.plot(r_list, sf_ratio[k][p].cpu().numpy(),
                color=col, linestyle=ls, marker=mk, markersize=3.5,
                markevery=1, label=lbl, alpha=0.85)

    ax.set_xscale('log', base=2)
    ax.set_yscale(yscale)
    if yscale == 'linear':
        ax.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))   # y ticks every 10
    shade_hybrid_uncompressed(ax)
    ax.set_xlabel(r'$r$', **label_kw)
    ax.grid(True, which='both', alpha=0.3)

# Panel labels a)/b)/c) with the plotted quantity, in the top-left of each panel.
panel_labels = [r'a) $S_4/S_2^2(r)$', r'b) $S_6/S_2^3(r)$', r'c) $S_8/S_2^4(r)$']
panel_label_x = [0.14, 0.24, 0.24]   # b)/c) shifted right to clear the TT standard r=2 point
for ax, lab, lx in zip(axes, panel_labels, panel_label_x):
    ax.text(lx, 0.96, lab, transform=ax.transAxes,
            ha='left', va='top', fontsize=9)

# Legends split across the panels (each panel contains the same artists):
# DNS/Galerkin/Wavelet in b), the two TT methods in c), and the annotation markers
# (PDF scales, TT hybrid - exact) in a).
handles, labels = axes[0].get_legend_handles_labels()
by_label = dict(zip(labels, handles))
annot_labels = ['PDF scales (Fig. 2)', 'TT hybrid - exact']
model_a = [signal_styles[k][0] for k in ['original', 'galerkin', 'wavelet_db32']]
model_b = [signal_styles[k][0] for k in ['standard_p4', 'hybrid_N12_p4_int64']]
legend_loc = ['center right', 'center right', 'center right']   # one per panel a)/b)/c)
legend_y   = [0.5, 0.65, 0.5]    # vertical centre of each legend box (axes fraction)
axes[0].legend([by_label[l] for l in annot_labels], annot_labels,
               loc=legend_loc[0], bbox_to_anchor=(1, legend_y[0]), handlelength=1.8)
axes[1].legend([by_label[l] for l in model_a], model_a,
               loc=legend_loc[1], bbox_to_anchor=(1, legend_y[1]), handlelength=1.8)
axes[2].legend([by_label[l] for l in model_b], model_b,
               loc=legend_loc[2], bbox_to_anchor=(1, legend_y[2]), handlelength=1.8)

# Same panel geometry as Fig. 1 (so panel a) matches Fig. 1 c)).
fig.tight_layout(rect=[0, 0.18, 1, 1])

# Save final figure
fig_png = os.path.join(fig_dir, 'figure7.png')
fig_pdf = os.path.join(fig_dir, 'figure7.pdf')
plt.savefig(fig_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_pdf, bbox_inches='tight')
plt.close()

print(f"Saved figure7.png and figure7.pdf in {fig_dir}")
print("Finished execution.")
