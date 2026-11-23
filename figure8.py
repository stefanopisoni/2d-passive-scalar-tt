"""Figure 8: is the sudden drop of the TT-hybrid E(k) set by the exact region or by the budget?

Energy spectra of the hybrid TT (p=4, seam-interpolated *_int64 fields, as in Figs. 1, 5):
  left column  - fixed N=12 (exact region: coarse blocks of 2^(L-N/2) = 64 px), varying
                 compression budget chi_std = 50 ... 1000 (hybrid at the same #params);
  right column - fixed budget chi_std = 100, varying N = 8, 12, 16
                 (coarse blocks of 256, 64, 16 px).
Top row: E(k). Bottom row: E_hybrid(k) / E_DNS(k).
Vertical markers: the exact-region boundary k = 2^L / 2^(L-N/2) = 2^(N/2) (block size taken
as a wavelength, as in the Fig. 1 shading), and the Galerkin cutoff at the same number of
parameters. If the drop moves with the budget at fixed N, the budget sets it; if it moves
with N at fixed budget, the exact region does.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
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
# PLOT_ONLY=True skips the averaging loop and re-plots from the saved
# figure8_data.pt in `stats_dir`. Set to False to recompute and overwrite it.
PLOT_ONLY = True
L = 12
path = cfg.DATA_DIR          # raw DNS snapshots (only needed if PLOT_ONLY=False)
hyb_dir = cfg.HYB_DIR        # precompressed hybrid TT fields
stats_dir = cfg.STATS_DIR
N_FIXED   = 12                         # left column: exact region fixed
CHI_LIST  = [50, 100, 200, 500, 1000]  # left column: budget (standard-TT bond dim. equivalent)
CHI_FIXED = 100                        # right column: budget fixed
N_LIST    = [8, 12, 16]                # right column: exact region (p1 = 2^N)
configs_to_average = list(range(59, 69))   # snapshots for which the N=8/16 fields exist
num_snaps = len(configs_to_average)

# Remaining bond dimension of each hybrid at chi_std = 100 (from generate_data.py, cf. Fig. 5).
BD_REMAINING_FIXED = {8: 101, 12: 49, 16: 3}

# Every (N, chi) hybrid variant needed by the two columns (N=12, chi=100 is shared).
variants = list(dict.fromkeys([(N_FIXED, chi) for chi in CHI_LIST] +
                              [(N, CHI_FIXED) for N in N_LIST]))

def k_exact(N):
    """Exact-region boundary in k: 2^L / block size, block size = 2^(L - N/2) px."""
    return 2**L / 2**(L - N // 2)

# ── Styles ──────────────────────────────────────────────────────────────────────
dns_style = ('DNS', 'black', '-')
# Budget scan: sequential colours, light -> dark with increasing budget.
chi_colors = {chi: mpl.cm.viridis(x) for chi, x in
              zip(CHI_LIST, np.linspace(0.0, 0.85, len(CHI_LIST))[::-1])}
# N scan: same colours as Fig. 5.
N_colors = {8: 'orange', 12: 'red', 16: 'darkred'}
hybrid_color = 'red'          # exact-region shading, as in Fig. 1

# ── Helpers ───────────────────────────────────────────────────────────────────
def read_config(config):
    filepath = os.path.join(path, f'theta.{config:05d}')
    theta32 = np.fromfile(filepath, dtype=np.float32)
    theta32 = theta32[1:-1].reshape(4096, -1)
    theta = theta32.view(np.float64)
    return theta

def first_k_below(k, ratio, thr):
    """Smallest k >= 1 at which E_hyb/E_DNS falls below `thr`."""
    idx = np.where((k >= 1) & (ratio < thr))[0]
    return float(k[idx[0]]) if len(idx) else float('nan')

if PLOT_ONLY:
    # ── Load previously computed statistics (skip the averaging loop) ──────────
    data_path = os.path.join(stats_dir, 'figure8_data.pt')
    print(f"PLOT_ONLY=True: loading {data_path} ...")
    _d = torch.load(data_path, weights_only=False)
    k_vals     = _d['k_vals']
    E_dns      = _d['E_dns']
    E_hyb      = _d['E_hyb']
    avg_cr     = _d['avg_cr']
    k_galerkin = _d['k_galerkin']
    print("Loaded figure8_data.pt.")
else:
    # Recompute path: needs spectral_tools + the raw/compressed data on disk.
    from spectral_tools import compute_isotropic_spectrum_2d, galerkin_truncate_2d
    E_dns = None
    E_hyb = {v: None for v in variants}
    cr_acc = {v: 0.0 for v in variants}
    k_vals = None

    print(f"Computing E(k) for {len(variants)} hybrid variants over {num_snaps} snapshots...")
    for config in configs_to_average:
        snap_str = f"{config:05d}"
        print(f"Processing snapshot {snap_str}...")
        f_orig = torch.from_numpy(read_config(config))
        k_vals, E = compute_isotropic_spectrum_2d(f_orig)
        E_dns = E if E_dns is None else E_dns + E

        for (N, chi) in variants:
            f_hyb = torch.load(os.path.join(
                hyb_dir, f'theta{snap_str}_hybrid_mps_p4_N{N}_bd_equiv{chi}_int64.tens'))
            cr_acc[(N, chi)] += float(torch.load(os.path.join(
                hyb_dir, f'theta{snap_str}_hybrid_mps_p4_N{N}_bd_equiv{chi}.compression')))
            _, E = compute_isotropic_spectrum_2d(f_hyb)
            E_hyb[(N, chi)] = E if E_hyb[(N, chi)] is None else E_hyb[(N, chi)] + E

    E_dns = E_dns / num_snaps
    E_hyb = {v: E / num_snaps for v, E in E_hyb.items()}
    avg_cr = {v: c / num_snaps for v, c in cr_acc.items()}

    # Galerkin cutoff at the same number of parameters as each hybrid. The cutoff
    # depends only on the parameter count (not on the field), so a zero field suffices.
    _zero = torch.zeros(2**L, 2**L, dtype=torch.float64)
    k_galerkin = {v: galerkin_truncate_2d(_zero, avg_cr[v] * 2**(2 * L))[2] for v in variants}

    # ── Save Statistics ───────────────────────────────────────────────────────────
    os.makedirs(stats_dir, exist_ok=True)
    torch.save({'k_vals': k_vals, 'E_dns': E_dns, 'E_hyb': E_hyb, 'avg_cr': avg_cr,
                'k_galerkin': k_galerkin, 'configs': configs_to_average},
               os.path.join(stats_dir, 'figure8_data.pt'))
    print(f"Saved figure8_data.pt in {stats_dir}")

    df_E = pd.DataFrame({'k': k_vals.cpu().numpy(), 'dns': E_dns.cpu().numpy()})
    for (N, chi), E in E_hyb.items():
        df_E[f'hybrid_N{N}_chi{chi}'] = E.cpu().numpy()
    df_E.to_csv(os.path.join(stats_dir, 'energy_spectra_hybrid_scan.csv'), index=False)
    print("Saved energy_spectra_hybrid_scan.csv")

# ── Summary: where each hybrid departs from the DNS ─────────────────────────────
k_np = k_vals.cpu().numpy()
# Ratio only where the DNS carries energy: past the DNS (dealiasing) cutoff E_DNS ~ 0
# and the ratio blows up.
E_dns_np = E_dns.cpu().numpy()
dns_has_energy = E_dns_np > 1e-10 * E_dns_np.max()
ratio = {v: np.where(dns_has_energy, E_hyb[v].cpu().numpy() / np.where(dns_has_energy, E_dns_np, 1.0),
                     np.nan) for v in variants}
rows = []
for (N, chi) in variants:
    rows.append({'N': N, 'chi_std': chi, 'rho_percent': 100 * avg_cr[(N, chi)],
                 'k_exact': k_exact(N), 'k_galerkin': k_galerkin[(N, chi)],
                 'k_ratio_below_0.9': first_k_below(k_np, ratio[(N, chi)], 0.9),
                 'k_ratio_below_0.5': first_k_below(k_np, ratio[(N, chi)], 0.5)})
df_summary = pd.DataFrame(rows)
print(df_summary.to_string(index=False, float_format=lambda x: f'{x:.3g}'))
os.makedirs(stats_dir, exist_ok=True)
df_summary.to_csv(os.path.join(stats_dir, 'figure8_summary.csv'), index=False)

# ── Plotting (2 Rows x 2 Columns) ─────────────────────────────────────────────
fig_dir = cfg.FIG_DIR
os.makedirs(fig_dir, exist_ok=True)

print("Generating figure8...")
# ── Layout (hand-tunable) ─────────────────────────────────────────────────────
FIG_WIDTH  = 6.5     # inches: \textwidth
FIG_HEIGHT = 4.4     # inches: two rows
E_YLIM     = (1e-2, None)    # E(k) panels, as Fig. 1 a)
RATIO_YLIM = (0, 1.2)        # ratio panels
# Panel-label position (axes fraction, top-right anchored), one per panel.
label_pos  = {'a': (0.97, 0.96), 'b': (0.97, 0.96), 'c': (0.97, 0.96), 'd': (0.97, 0.96)}
# Legend location, one per panel that carries a legend.
legend_loc = {'a': 'lower left', 'c': 'lower left', 'd': 'lower left'}

fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300, sharex=True)
ax_a, ax_b = axes[0]
ax_c, ax_d = axes[1]
legend_kw = dict(handlelength=1.8, labelspacing=0.3, borderpad=0.3, handletextpad=0.5)

def shade_exact(ax, N):
    """Shade the exact region k <= k_exact(N) (call after all curves are drawn)."""
    xlim = ax.get_xlim()
    ax.axvspan(xlim[0], k_exact(N), color=hybrid_color, alpha=0.08, zorder=0)
    ax.set_xlim(xlim)

lbl, col, ls = dns_style
for ax in (ax_a, ax_b):
    ax.loglog(k_np, E_dns.cpu().numpy(), color=col, linestyle=ls, linewidth=1.2,
              label=lbl, zorder=3)
for ax in (ax_c, ax_d):
    ax.axhline(1.0, color='black', linewidth=0.8, alpha=0.6, zorder=1)
    ax.axhline(0.5, color='grey', linewidth=0.6, linestyle='--', alpha=0.6, zorder=1)

# Left column: fixed N, varying budget.
for chi in CHI_LIST:
    v = (N_FIXED, chi)
    c = chi_colors[chi]
    lab = rf'$\chi={chi}$ ($\rho={100 * avg_cr[v]:.2f}\%$)'
    ax_a.loglog(k_np, E_hyb[v].cpu().numpy(), color=c, linewidth=1.0, alpha=0.9, label=lab)
    ax_c.semilogx(k_np, ratio[v], color=c, linewidth=1.0, alpha=0.9, label=lab)
    ax_c.axvline(k_galerkin[v], color=c, linestyle=':', linewidth=1.0, alpha=0.9)

# Right column: fixed budget, varying N.
for N in N_LIST:
    v = (N, CHI_FIXED)
    c = N_colors[N]
    lab = rf'$L_h={N // 2}$ ($\tilde{{\chi}}={BD_REMAINING_FIXED[N]}$)'
    ax_b.loglog(k_np, E_hyb[v].cpu().numpy(), color=c, linestyle='-.', linewidth=1.0,
                alpha=0.9, label=lab)
    ax_d.semilogx(k_np, ratio[v], color=c, linestyle='-.', linewidth=1.0, alpha=0.9, label=lab)
    for ax in (ax_b, ax_d):
        ax.axvline(k_exact(N), color=c, linestyle='--', linewidth=1.0, alpha=0.7)
    ax_d.axvline(k_galerkin[v], color=c, linestyle=':', linewidth=1.0, alpha=0.9)

for ax in (ax_a, ax_b):
    ax.set_ylim(*E_YLIM)
for ax in (ax_c, ax_d):
    ax.set_ylim(*RATIO_YLIM)
    ax.set_xlabel(r'$k$')
for ax in axes.ravel():
    ax.grid(True, which='both', alpha=0.3)
ax_a.set_xlim(left=1)      # shared x: k >= 1 (k=0 is dropped on the log axis)
for ax in (ax_a, ax_c):
    shade_exact(ax, N_FIXED)

# Panel labels
panel_text = {
    'a': rf'a) $E(k)$, $L_h={N_FIXED // 2}$',
    'b': rf'b) $E(k)$, $\chi={CHI_FIXED}$',
    'c': rf'c) $E/E_{{\mathrm{{DNS}}}}$, $L_h={N_FIXED // 2}$',
    'd': rf'd) $E/E_{{\mathrm{{DNS}}}}$, $\chi={CHI_FIXED}$',
}
for key, ax in zip('abcd', (ax_a, ax_b, ax_c, ax_d)):
    lx, ly = label_pos[key]
    ax.text(lx, ly, panel_text[key], transform=ax.transAxes, ha='right', va='top',
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.0), zorder=5)

# Legends: curves in the ratio panels, marker meanings in a).
ax_c.legend(loc=legend_loc['c'], **legend_kw)
ax_d.legend(loc=legend_loc['d'], **legend_kw)
annot_handles = [
    Patch(facecolor=hybrid_color, alpha=0.08, label=rf'head scales ($L_h={N_FIXED // 2}$)'),
    Line2D([], [], color='grey', linestyle='--', label=r'head edge $k=2^{L_h}$'),
    Line2D([], [], color='grey', linestyle=':', label=r'Galerkin cutoff, same $N_{\mathrm{par}}$'),
]
ax_a.legend(handles=[ax_a.get_lines()[0]] + annot_handles, loc=legend_loc['a'], **legend_kw)

fig.tight_layout()

fig_png = os.path.join(fig_dir, 'figure8.png')
fig_pdf = os.path.join(fig_dir, 'figure8.pdf')
plt.savefig(fig_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_pdf, bbox_inches='tight')
plt.close()

print(f"Saved figure8.png and figure8.pdf in {fig_dir}")
print("Finished execution.")
