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
# \textwidth = 6.5 in and \columnwidth = 3.125 in. Figures are authored at their
# true display width with Computer-Modern serif fonts and journal-sized text, so
# that nothing is shrunk on inclusion and labels/legends/titles match the body
# text. Uses LaTeX rendering when a latex+dvipng pipeline is available, and falls
# back to matplotlib's Computer-Modern mathtext otherwise.
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
# figure2_data.pt in `stats_dir`. Set to False to recompute and overwrite it.
PLOT_ONLY = True
L = 12
path = cfg.DATA_DIR          # raw DNS snapshots (only needed if PLOT_ONLY=False)
std_dir = cfg.STD_DIR        # precompressed standard TT fields
hyb_dir = cfg.HYB_DIR        # precompressed hybrid TT fields
stats_dir = cfg.STATS_DIR
downsample = 1
BD_MAX = 100
r_pdf_list  = [2, 4, 8]                    # Scales at which the PDF panels are drawn
r_all       = sorted(set(r_pdf_list))      # all scales whose PDFs we accumulate
bins = torch.arange(-20, 20, 0.1)
configs_to_average = list(range(59, 109)) + list(range(509, 559))
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

if PLOT_ONLY:
    # ── Load previously computed statistics (skip the heavy averaging loop) ─────
    data_path = os.path.join(stats_dir, 'figure2_data.pt')
    print(f"PLOT_ONLY=True: loading {data_path} ...")
    _d = torch.load(data_path, weights_only=False)
    bin_centers = _d['bin_centers']
    pdf_acc     = _d['pdf_acc']
    r_pdf_list  = _d['r_pdf_list']
    bins        = _d['bins']
    avg_cr      = _d['avg_cr']
    print(f"Loaded figure2_data.pt (avg CR = {avg_cr:.4f}).")
else:
    # Recompute path: needs spectral_tools + the raw/compressed data on disk.
    from spectral_tools import galerkin_truncate_2d, wavelet_truncate_2d_pywt
    # ── Streaming two-pass over snapshots (flat memory, scales to many configs) ──
    # For every (signal, scale r) the increments are pooled over ALL snapshots and
    # normalized by a single pooled std (unbiased, == torch.std over the pool).
    # Pass 1 accumulates sum / sum-of-squares / count -> pooled std.
    # Pass 2 re-reads snapshots, normalizes, and accumulates raw histogram counts.
    # Only one snapshot's fields live in memory at a time. NOTE: each snapshot is
    # processed twice, so the Galerkin and db32 truncations are recomputed in both passes.
    bins64 = bins.to(dtype=torch.float64)

    def build_fields(config):
        """Return (std_p4_cr, {signal: field}) for one snapshot."""
        snap_str = f"{config:05d}"
        f_orig = torch.from_numpy(read_config(config))
        std_p4_cr = torch.load(
            os.path.join(std_dir, f'theta{snap_str}_standard_mps_bd{BD_MAX}_p4.compression'))
        dof_target = std_p4_cr * (2**24)
        f_galerkin, _, _ = galerkin_truncate_2d(f_orig, dof_target)
        f_wavelet_db32 = wavelet_truncate_2d_pywt(f_orig, dof_target, wavelet='db32')
        f_std_p4 = torch.load(
            os.path.join(std_dir, f'theta{snap_str}_standard_mps_bd{BD_MAX}_p4.tens'))
        f_hybrid_int64 = torch.load(
            os.path.join(hyb_dir, f'theta{snap_str}_hybrid_mps_p4_N12_bd_equiv{BD_MAX}_int64.tens'))
        return std_p4_cr, {
            'original':            f_orig,
            'galerkin':            f_galerkin,
            'wavelet_db32':        f_wavelet_db32,
            'standard_p4':         f_std_p4,
            'hybrid_N12_p4_int64': f_hybrid_int64,
        }

    # ── Pass 1: pooled std per (signal, r) ──────────────────────────────────────
    acc_sum   = {k: {r: 0.0 for r in r_all} for k in signal_keys}
    acc_sumsq = {k: {r: 0.0 for r in r_all} for k in signal_keys}
    acc_count = {k: {r: 0   for r in r_all} for k in signal_keys}
    cr_acc = 0.0

    print(f"Pass 1/2: pooled std for BD_MAX={BD_MAX} across {num_snaps} snapshots...")
    for config in configs_to_average:
        print(f"  [pass 1] snapshot {config:05d}...")
        std_p4_cr, fields = build_fields(config)
        cr_acc += std_p4_cr
        for k, f in fields.items():
            for r in r_all:
                inc = compute_increments(f, axis=1, r=r, stride=downsample).ravel()
                acc_sum[k][r]   += inc.sum().item()
                acc_sumsq[k][r] += inc.pow(2).sum().item()
                acc_count[k][r] += inc.numel()

    avg_cr = cr_acc / num_snaps
    std = {k: {} for k in signal_keys}
    for k in signal_keys:
        for r in r_all:
            n = acc_count[k][r]
            var = (acc_sumsq[k][r] - acc_sum[k][r] ** 2 / n) / (n - 1)   # unbiased == torch.std
            std[k][r] = var ** 0.5

    # ── Pass 2: histograms with the pooled std ──────────────────────────────────
    hist_counts = {k: {r: torch.zeros(len(bins64) - 1, dtype=torch.float64) for r in r_all}
                   for k in signal_keys}
    print("Pass 2/2: histograms with the pooled std...")
    for config in configs_to_average:
        print(f"  [pass 2] snapshot {config:05d}...")
        _, fields = build_fields(config)
        for k, f in fields.items():
            for r in r_all:
                inc = compute_increments(f, axis=1, r=r, stride=downsample).ravel()
                s = std[k][r]
                if s > 0:
                    inc = inc / s
                hist_counts[k][r] += torch.histogram(inc.cpu(), bins=bins64, density=False).hist.to(torch.float64)

    # Convert pooled raw counts to a probability density (integral over range = 1)
    widths = bins64[1:] - bins64[:-1]
    pdf_acc = {k: {r: hist_counts[k][r] / (hist_counts[k][r].sum() * widths) for r in r_all}
               for k in signal_keys}
    print(f"Average compression ratio: {avg_cr:.4f}")

    # ── Save Statistics ───────────────────────────────────────────────────────────
    os.makedirs(stats_dir, exist_ok=True)

    bin_centers = (0.5 * (bins[:-1] + bins[1:])).cpu().numpy()

    # 1. Save consolidated PyTorch file
    pytorch_data = {
        'bin_centers': bin_centers,
        'pdf_acc': pdf_acc,
        'r_pdf_list': r_pdf_list,
        'bins': bins,
        'avg_cr': avg_cr
    }
    torch.save(pytorch_data, os.path.join(stats_dir, 'figure2_data.pt'))
    print(f"Saved figure2_data.pt in {stats_dir}")

    # 2. Save Increment PDFs to CSV
    df_pdf = pd.DataFrame({'bin_center': bin_centers})
    for r in r_all:
        for k in signal_keys:
            df_pdf[f'{k}_r{r}'] = pdf_acc[k][r].cpu().numpy()
    df_pdf.to_csv(os.path.join(stats_dir, 'increments_pdf.csv'), index=False)
    print("Saved increments_pdf.csv")

# ── Plotting (1 Row x 3 Columns) ──────────────────────────────────────────────
fig_dir = cfg.FIG_DIR
os.makedirs(fig_dir, exist_ok=True)

print("Generating increment PDF figure...")
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.0), dpi=300, sharey=True)

# Per-element keyword args (sizes/weights now come from rcParams above)
title_kw = dict(pad=4)
label_kw = dict()
legend_kw = dict(loc='best')

# Generate subplots for r = 2, 4, 8
for idx, r in enumerate(r_pdf_list):
    ax = axes[idx]
    for k in signal_keys:
        lbl, col, ls, _ = signal_styles[k]
        # Since it is a PDF with 399 points, plot as lines without markers for clean aesthetics
        ax.semilogy(bin_centers, pdf_acc[k][r].cpu().numpy(),
                    color=col, linestyle=ls, linewidth=1.0, label=lbl, alpha=0.85)

    ax.set_xlabel(r'$\delta_r\theta / \sigma_{\delta_r\theta}$', **label_kw)
    # if idx == 0:
        # Shared y-axis: write its quantity horizontally at the top instead of as a
        # rotated left label, freeing horizontal space.
        # ax.set_title('Probability density', loc='left', **title_kw)
    ax.set_xlim(-20, 20)
    # Top a little above 10^0 so the peak doesn't touch the frame; keep the 10^0 tick.
    ax.set_ylim(1e-7, 2e0)
    ax.set_yticks([10.0**e for e in range(-6, 1, 2)])   # even decades: 10^-6,-4,-2,0
    ax.grid(True, which='both', alpha=0.3)
    # Panel label a)/b)/c) with its separation r (cf. Fig. 4).
    ax.text(0.04, 0.94, rf'{"abc"[idx]}) $r={r}$', transform=ax.transAxes,
            ha='left', va='top')

# Legend split across the three panels, in the empty region under the peaked PDFs
# (the curves are common to all three panels): Galerkin/Wavelet in panel 1, the two
# TT methods in panel 2, and DNS alone in panel 3.
handles, labels = axes[0].get_legend_handles_labels()
axes[0].legend(handles[1:3], labels[1:3], loc='lower center', handlelength=1.8)
axes[1].legend(handles[3:], labels[3:], loc='lower center', handlelength=1.8)
axes[2].legend(handles[:1], labels[:1], loc='lower center', handlelength=1.8)
fig.tight_layout()

# Save final figure
fig_png = os.path.join(fig_dir, 'figure2.png')
fig_pdf = os.path.join(fig_dir, 'figure2.pdf')
plt.savefig(fig_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_pdf, bbox_inches='tight')
plt.close()

print(f"Saved figure2.png and figure2.pdf in {fig_dir}")

print("Finished execution.")
