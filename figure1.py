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
# figure1_data.pt in `stats_dir`. Set to False to recompute and overwrite it.
PLOT_ONLY = True
L = 12
path = cfg.DATA_DIR          # raw DNS snapshots (only needed if PLOT_ONLY=False)
std_dir = cfg.STD_DIR        # precompressed standard TT fields
hyb_dir = cfg.HYB_DIR        # precompressed hybrid TT fields
stats_dir = cfg.STATS_DIR
downsample = 1
BD_MAX = 100
p_list = [2, 4]  # We need p=2 and p=4 for Flatness S4/S2^2
r_list = [2**i for i in range(12)]
r_pdf_marks = [2, 4, 8]   # scales at which figure2.py plots the increment PDFs
n_hybrid_uncompressed = 6  # the hybrid MPS leaves the 6 biggest scales uncompressed
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
    data_path = os.path.join(stats_dir, 'figure1_data.pt')
    print(f"PLOT_ONLY=True: loading {data_path} ...")
    _d = torch.load(data_path, weights_only=False)
    k_vals   = _d['k_vals']
    E_acc    = _d['E_acc']
    r_list   = _d['r_list']
    sf_acc   = _d['sf_acc']
    flatness = _d['flatness']
    avg_cr   = _d['avg_cr']
    print(f"Loaded figure1_data.pt (avg CR = {avg_cr:.4f}).")
else:
    # Recompute path: needs spectral_tools + the raw/compressed data on disk.
    from spectral_tools import (galerkin_truncate_2d, wavelet_truncate_2d_pywt,
                                 compute_isotropic_spectrum_2d)
    # ── Accumulators ────────────────────────────────────────────────────────────
    sf_acc = {k: {p: torch.zeros(len(r_list)) for p in p_list} for k in signal_keys}
    E_acc = {k: None for k in signal_keys}
    k_vals = None
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

        # Accumulate energy spectrum and structure functions for each signal
        for k, f in fields.items():
            # Structure functions
            sf = compute_structure_functions([f], r_list, p_list=p_list, axis=1, stride=downsample)
            for p in p_list:
                sf_acc[k][p] += sf[p]

            # Energy spectrum
            curr_k, E = compute_isotropic_spectrum_2d(f)
            if k_vals is None:
                k_vals = curr_k
            if E_acc[k] is None:
                E_acc[k] = torch.zeros_like(E)
            E_acc[k] += E

    # ── Finalize Averages ──────────────────────────────────────────────────────────
    avg_cr = cr_acc / num_snaps
    print(f"Finalizing averages. Average compression ratio (std p=4, BD={BD_MAX}): {avg_cr:.4f}")

    for k in signal_keys:
        for p in p_list:
            sf_acc[k][p] /= num_snaps
        E_acc[k] /= num_snaps

    # Compute Flatness F(r) = S_4(r) / S_2(r)^2
    flatness = {}
    for k in signal_keys:
        flatness[k] = sf_acc[k][4] / (sf_acc[k][2] ** 2)

    # ── Save Statistics ───────────────────────────────────────────────────────────
    os.makedirs(stats_dir, exist_ok=True)

    # 1. Save consolidated PyTorch file
    pytorch_data = {
        'k_vals': k_vals,
        'E_acc': E_acc,
        'r_list': r_list,
        'sf_acc': sf_acc,
        'flatness': flatness,
        'avg_cr': avg_cr
    }
    torch.save(pytorch_data, os.path.join(stats_dir, 'figure1_data.pt'))
    print(f"Saved figure1_data.pt in {stats_dir}")

    # 2. Save Energy Spectra to CSV
    df_energy = pd.DataFrame({'k': k_vals.cpu().numpy()})
    for k in signal_keys:
        df_energy[k] = E_acc[k].cpu().numpy()
    df_energy.to_csv(os.path.join(stats_dir, 'energy_spectra.csv'), index=False)
    print("Saved energy_spectra.csv")

    # 3. Save Structure Functions (S4) to CSV
    df_sf = pd.DataFrame({'r': r_list})
    for k in signal_keys:
        df_sf[f'{k}_S2'] = sf_acc[k][2].cpu().numpy()
        df_sf[f'{k}_S4'] = sf_acc[k][4].cpu().numpy()
    df_sf.to_csv(os.path.join(stats_dir, 'structure_functions.csv'), index=False)
    print("Saved structure_functions.csv")

    # 4. Save Flatness to CSV
    df_flat = pd.DataFrame({'r': r_list})
    for k in signal_keys:
        df_flat[k] = flatness[k].cpu().numpy()
    df_flat.to_csv(os.path.join(stats_dir, 'flatness.csv'), index=False)
    print("Saved flatness.csv")

# ── Plotting (1 Row x 3 Columns) ──────────────────────────────────────────────
fig_dir = cfg.FIG_DIR
os.makedirs(fig_dir, exist_ok=True)

print("Generating comparison figure...")
fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4), dpi=300)

# Per-element keyword args (sizes/weights now come from rcParams above)
title_kw = dict(pad=4)
label_kw = dict()
legend_kw = dict(loc='best')

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

# 1. Energy Spectra Plot
ax_e = axes[0]
# Fewer markers on the smooth curves (they were too dense); Galerkin keeps its
# denser spacing so its inertial-range triangles stay as before.
me_default  = max(1, len(k_vals)//7)
me_galerkin = max(1, len(k_vals)//15)
for k in signal_keys:
    lbl, col, ls, mk = signal_styles[k]
    me = me_galerkin if k == 'galerkin' else me_default
    ax_e.loglog(k_vals.cpu().numpy(), E_acc[k].cpu().numpy(),
                color=col, linestyle=ls, marker=mk, markersize=3.5,
                markevery=me, label=lbl, alpha=0.85)

# Reference -5/3 line (removed from the E(k) panel)
# k_ref = k_vals[10:100].cpu().numpy()
# E_ref = E_acc['original'][10].item() * (k_ref / k_ref[0])**(-5/3)
# ax_e.loglog(k_ref, E_ref, 'k--', linewidth=1.0, label=r'$k^{-5/3}$ fit')

ax_e.set_xlabel(r'$k$', **label_kw)
# ax_e.set_title(r'E($k$)', **title_kw)
ax_e.set_ylim(1e-2, None)
ax_e.grid(True, which='both', alpha=0.3)

# The Galerkin spectrum drops near-vertically at its spectral cutoff (k~265): the
# next point past the cutoff is numerically zero (~1e-28), so the line plunges
# straight to the bottom with no data points on the visible segment to carry
# markers. Since the whole drop sits at essentially one k, stack a column of
# triangles at that cutoff k, from the top of the drop down to the plot floor.
_g_lbl, _g_col, _g_ls, _g_mk = signal_styles['galerkin']
_kg = k_vals.cpu().numpy()
_Eg = E_acc['galerkin'].cpu().numpy()
_ylo = ax_e.get_ylim()[0]
_gvis = np.where((_Eg >= _ylo) & (_kg > 0))[0]
if len(_gvis):
    _kc = _kg[_gvis[-1]]                       # cutoff k (last above-floor point)
    _ytop = _Eg[_gvis[-4:]].max()             # top of the near-vertical drop
    _ycol = np.geomspace(_ylo, _ytop, 4)      # triangles from floor up to the top
    ax_e.loglog(np.full_like(_ycol, _kc), _ycol, color=_g_col, linestyle='none',
                marker=_g_mk, markersize=3.5, alpha=0.85)

# 2. Structure Function Order 4 Plot
ax_sf = axes[1]
for i, r in enumerate(r_pdf_marks):
    ax_sf.axvline(r, color='grey', linestyle='-', linewidth=2.5, alpha=0.25, zorder=0,
                  label='PDF scales (Fig. 2)' if i == 0 else None)
for k in signal_keys:
    lbl, col, ls, mk = signal_styles[k]
    ax_sf.loglog(r_list, sf_acc[k][4].cpu().numpy(),
                 color=col, linestyle=ls, marker=mk, markersize=3.5,
                 markevery=1, label=lbl, alpha=0.85)

ax_sf.set_xscale('log', base=2)   # power-of-2 x ticks, matching the flatness panel
shade_hybrid_uncompressed(ax_sf)
ax_sf.set_xlabel(r'$r$', **label_kw)
# ax_sf.set_title(r'$S_4(r)$', **title_kw)
ax_sf.grid(True, which='both', alpha=0.3)

# 3. Flatness Plot (Log x, Linear y)
ax_fl = axes[2]
for i, r in enumerate(r_pdf_marks):
    ax_fl.axvline(r, color='grey', linestyle='-', linewidth=2.5, alpha=0.25, zorder=0,
                  label='PDF scales (Fig. 2)' if i == 0 else None)
for k in signal_keys:
    lbl, col, ls, mk = signal_styles[k]
    ax_fl.plot(r_list, flatness[k].cpu().numpy(),
               color=col, linestyle=ls, marker=mk, markersize=3.5,
               markevery=1, label=lbl, alpha=0.85)

ax_fl.set_xscale('log', base=2)
ax_fl.yaxis.set_major_locator(mpl.ticker.MultipleLocator(10))   # y ticks every 10
shade_hybrid_uncompressed(ax_fl)
ax_fl.set_xlabel(r'$r$', **label_kw)
# ax_fl.set_title(r'$S_4 / S_2^2$', **title_kw)
ax_fl.grid(True, which='both', alpha=0.3)

# Panel labels a)/b)/c) with the plotted quantity, in the top-left of each panel.
panel_labels = [r'a) $E(k)$', r'b) $S_4(r)$', r'c) $S_4/S_2^2(r)$']
panel_label_x = [0.04, 0.14, 0.14]
for ax, lab, lx in zip(axes, panel_labels, panel_label_x):
    ax.text(lx, 0.96, lab, transform=ax.transAxes,
            ha='left', va='top', fontsize=9)

# In-panel legends instead of a single shared legend below the row:
#  - the model curves are split across two corners of the E(k) panel so neither
#    box swamps the data: TT standard/TT hybrid in the top-right, and the remaining
#    three (DNS, Galerkin, Wavelet) in the bottom-left. The E(k) panel contains
#    exactly these artists, in signal_keys order.
#  - the annotation markers (PDF scales, TT hybrid - exact) go in the bottom-right
#    of the S_4 panel.
e_handles, e_labels = ax_e.get_legend_handles_labels()
top_idx = [3, 4]            # TT standard, TT hybrid
bot_idx = [0, 1, 2]         # DNS, Galerkin, Wavelet
leg_top = ax_e.legend([e_handles[i] for i in top_idx], [e_labels[i] for i in top_idx],
                      loc='upper right', handlelength=1.8)
ax_e.add_artist(leg_top)
ax_e.legend([e_handles[i] for i in bot_idx], [e_labels[i] for i in bot_idx],
            loc='lower left', handlelength=1.8)

sf_handles, sf_labels = ax_sf.get_legend_handles_labels()
annot_labels = ['PDF scales (Fig. 2)', 'TT hybrid - exact']
sel = [(h, l) for h, l in zip(sf_handles, sf_labels) if l in annot_labels]
sel.sort(key=lambda hl: annot_labels.index(hl[1]))
ax_sf.legend([h for h, _ in sel], [l for _, l in sel],
             loc='lower right', handlelength=1.8)

fig.tight_layout(rect=[0, 0.18, 1, 1])

# Save final figure
fig_png = os.path.join(fig_dir, 'figure1.png')
fig_pdf = os.path.join(fig_dir, 'figure1.pdf')
plt.savefig(fig_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_pdf, bbox_inches='tight')
plt.close()

print(f"Saved figure1.png and figure1.pdf in {fig_dir}")
print("Finished execution.")
