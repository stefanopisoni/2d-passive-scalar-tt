"""Figure 9: distance of each method's increment PDF from the DNS PDF, as a function of r.

The PDFs are computed exactly as in Fig. 2 (figure2.py): horizontal increments pooled over
the snapshots, each signal normalized by its own pooled std (two-pass, unbiased), bins of
width 0.1 on [-20, 20) std (same bins as Fig. 2). The difference from Fig. 2: all scales
r = 2^0 ... 2^11 instead of r = 2, 4, 8. The fraction of increments outside the histogram
range is written to the CSV. The histograms are cached in statistics/figure9_data.pt;
figure2's cache is not touched.

For each compressed method and scale r (x = normalized increment):
  W_1   - 1-Wasserstein (earth mover's) distance, sum_i |F_m(x_i) - F_DNS(x_i)| dx, with
          F the CDF of the binned distribution. Exact for the binned distributions; in
          units of the normalized increment (each signal's own std).
  D_JS  - Jensen-Shannon distance sqrt(JSD), JSD = (KL(P||M) + KL(Q||M)) / 2 with
          M = (P + Q) / 2, base-2 logarithm, so D_JS lies in [0, 1].
Both are dominated by where the probability mass is (the core of the PDF); the far tails,
which carry little mass, contribute little. The second row therefore repeats both metrics
on the tails alone: the distribution of |x| conditional on |x| > TAIL_X (the two tails
folded together), renormalized to unit mass for each signal. This compares the SHAPE of
the tails; how much mass each signal puts there, P(|x| > TAIL_X), is written to the CSV.
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

import config as cfg
# spectral_tools (and the raw/compressed data) are only needed to RECOMPUTE the
# cached histograms; imported lazily inside the PLOT_ONLY=False branch below.

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
# PLOT_ONLY=True skips the heavy two-pass loop and re-plots from the saved
# figure9_data.pt in `stats_dir`. Set to False to recompute and overwrite it.
PLOT_ONLY = True
L = 12
path = cfg.DATA_DIR          # raw DNS snapshots (only needed if PLOT_ONLY=False)
std_dir = cfg.STD_DIR        # precompressed standard TT fields
hyb_dir = cfg.HYB_DIR        # precompressed hybrid TT fields
stats_dir = cfg.STATS_DIR
downsample = 1
BD_MAX = 100
r_all = [2**i for i in range(L)]           # all scales 1 ... 2048
r_pdf_marks = [2, 4, 8]                    # scales of the Fig. 2 PDF panels
n_hybrid_uncompressed = 6                  # the hybrid MPS leaves the 6 biggest scales uncompressed
bins = torch.arange(-20, 20, 0.1).to(dtype=torch.float64)   # same bins as Fig. 2
TAIL_X = 3.0                 # tail region |x| > TAIL_X (in units of each signal's std)
# configs_to_average = list(range(59, 109)) + list(range(509, 559))   # full set (as Fig. 2)
# Test run: 20 snapshots, every 5th of each block so they are not consecutive/correlated.
configs_to_average = list(range(59, 109, 5)) + list(range(509, 559, 5))
num_snaps = len(configs_to_average)

# ── Signal catalogue (same styles as Fig. 2) ──────────────────────────────────
signal_styles = {
    'original':            ('DNS',                     'black',      '-',   'o'),
    'galerkin':            ('Galerkin',                'green',      ':',   'v'),
    'wavelet_db32':        ('Wavelet (db32)',          'purple',     '--',  'X'),
    'standard_p4':         ('TT standard',             'steelblue',  '--',  's'),
    'hybrid_N12_p4_int64': ('TT hybrid',               'red',        '-.',  'D'),
}
signal_keys = list(signal_styles.keys())
method_keys = [k for k in signal_keys if k != 'original']

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
    # ── Load previously computed histograms (skip the two-pass loop) ────────────
    data_path = os.path.join(stats_dir, 'figure9_data.pt')
    print(f"PLOT_ONLY=True: loading {data_path} ...")
    _d = torch.load(data_path, weights_only=False)
    pdf_acc     = _d['pdf_acc']
    r_all       = _d['r_all']
    bins        = _d['bins']
    outside     = _d['outside']
    avg_cr      = _d['avg_cr']
    print(f"Loaded figure9_data.pt ({len(_d['configs'])} snapshots, avg CR = {avg_cr:.4f}).")
else:
    # Recompute path: needs spectral_tools + the raw/compressed data on disk.
    from spectral_tools import galerkin_truncate_2d, wavelet_truncate_2d_pywt
    # ── Streaming two-pass over snapshots, as in figure2.py ──────────────────────
    # Pass 1 accumulates sum / sum-of-squares / count -> pooled std per (signal, r).
    # Pass 2 re-reads snapshots, normalizes, and accumulates raw histogram counts.

    def build_fields(config):
        """Return (std_p4_cr, {signal: field}) for one snapshot (same as figure2.py)."""
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
    hist_counts = {k: {r: torch.zeros(len(bins) - 1, dtype=torch.float64) for r in r_all}
                   for k in signal_keys}
    n_outside = {k: {r: 0 for r in r_all} for k in signal_keys}
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
                h = torch.histogram(inc.cpu(), bins=bins, density=False).hist.to(torch.float64)
                hist_counts[k][r] += h
                n_outside[k][r] += inc.numel() - int(h.sum().item())

    # Pooled raw counts -> probability density (integral over the range = 1), as Fig. 2.
    widths_t = bins[1:] - bins[:-1]
    pdf_acc = {k: {r: hist_counts[k][r] / (hist_counts[k][r].sum() * widths_t) for r in r_all}
               for k in signal_keys}
    outside = {k: {r: n_outside[k][r] / acc_count[k][r] for r in r_all} for k in signal_keys}
    print(f"Average compression ratio: {avg_cr:.4f}")

    # ── Save Statistics ───────────────────────────────────────────────────────────
    os.makedirs(stats_dir, exist_ok=True)
    torch.save({'pdf_acc': pdf_acc, 'r_all': r_all, 'bins': bins, 'std': std,
                'outside': outside, 'avg_cr': avg_cr, 'configs': configs_to_average},
               os.path.join(stats_dir, 'figure9_data.pt'))
    print(f"Saved figure9_data.pt in {stats_dir}")

# ── Distances between binned distributions ────────────────────────────────────
def bin_probabilities(pdf, widths):
    """Probability per bin from a density histogram (renormalized to sum to 1)."""
    p = np.asarray(pdf, dtype=np.float64) * widths
    return p / p.sum()

def tail_probabilities(p, centers, x_min):
    """Distribution of |x| conditional on |x| > x_min, from bin probabilities `p` at bin
    centres `centers` (both tails folded onto the |x| bins). Returns (|x| grid, probs, mass)."""
    a = np.round(np.abs(centers), 3)
    sel = a > x_min
    grid = np.unique(a[sel])
    q = np.zeros(len(grid))
    np.add.at(q, np.searchsorted(grid, a[sel]), p[sel])
    mass = q.sum()
    return grid, q / mass, float(mass)

def wasserstein_1(p, q, widths):
    """1-Wasserstein distance between two distributions on the same uniform bins."""
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))) * widths[0])

def jensen_shannon_distance(p, q):
    """sqrt of the base-2 Jensen-Shannon divergence; in [0, 1]."""
    m = 0.5 * (p + q)
    def kl(a, b):
        s = a > 0                      # 0 log 0 = 0; b >= a/2 > 0 wherever a > 0
        return np.sum(a[s] * np.log2(a[s] / b[s]))
    jsd = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return float(np.sqrt(max(jsd, 0.0)))

bins_np = bins.cpu().numpy().astype(np.float64)
widths  = np.diff(bins_np)
centers = 0.5 * (bins_np[:-1] + bins_np[1:])
assert np.allclose(widths, widths[0], rtol=1e-4), "W_1 formula assumes uniform bins"

def distances(key_m, key_ref, r):
    p_m   = bin_probabilities(pdf_acc[key_m][r].cpu().numpy(), widths)
    p_ref = bin_probabilities(pdf_acc[key_ref][r].cpu().numpy(), widths)
    _, t_m, mass_m     = tail_probabilities(p_m, centers, TAIL_X)
    _, t_ref, mass_ref = tail_probabilities(p_ref, centers, TAIL_X)
    return {'W1': wasserstein_1(p_m, p_ref, widths),
            'JS_distance': jensen_shannon_distance(p_m, p_ref),
            'W1_tail': wasserstein_1(t_m, t_ref, widths),
            'JS_distance_tail': jensen_shannon_distance(t_m, t_ref),
            'tail_mass': mass_m, 'tail_mass_ref': mass_ref,
            'fraction_outside_range': outside[key_m][r]}

rows = []
for r in r_all:
    for k in method_keys:
        rows.append({'method': k, 'r': r, **distances(k, 'original', r)})
df = pd.DataFrame(rows)
print(df.to_string(index=False, float_format=lambda x: f'{x:.4g}'))
os.makedirs(stats_dir, exist_ok=True)
df.to_csv(os.path.join(stats_dir, 'figure9_pdf_distances.csv'), index=False)
print("Saved figure9_pdf_distances.csv")

# ── Plotting (2 Rows x 2 Columns) ─────────────────────────────────────────────
fig_dir = cfg.FIG_DIR
os.makedirs(fig_dir, exist_ok=True)

print("Generating figure9...")
# ── Layout (hand-tunable) ─────────────────────────────────────────────────────
FIG_WIDTH  = 6.5     # inches: \textwidth
FIG_HEIGHT = 4.4     # inches: two rows
YLIM       = {'W1': (0, 0.10), 'JS_distance': (0, 0.11),         # headroom for the panel labels
              'W1_tail': (0, 0.72), 'JS_distance_tail': (0, 0.165)}
# Panel-label position (axes fraction, top-left anchored), one per panel.
label_pos  = {'a': (0.04, 0.96), 'b': (0.04, 0.96), 'c': (0.04, 0.96), 'd': (0.04, 0.96)}
# Legend split across the panels: TT methods in a), Galerkin/Wavelet in b).
legend_loc = {'a': 'upper right', 'b': 'upper right'}
legend_kw  = dict(handlelength=1.8, labelspacing=0.3, borderpad=0.3, handletextpad=0.5)

fig, axes = plt.subplots(2, 2, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300, sharex=True)
_tail = rf'$|\delta_r\theta|/\sigma_{{\delta_r\theta}} > {TAIL_X:g}$'
panels = [('a', 'W1', r'a) $W_1$'),
          ('b', 'JS_distance', r'b) $D_{\mathrm{JS}}$'),
          ('c', 'W1_tail', r'c) $W_1$, ' + _tail),
          ('d', 'JS_distance_tail', r'd) $D_{\mathrm{JS}}$, ' + _tail)]

# Shaded region marking the 6 biggest scales that the TT hybrid leaves uncompressed (Fig. 1).
hybrid_color = signal_styles['hybrid_N12_p4_int64'][1]
hybrid_shade_left = sorted(r_all)[-n_hybrid_uncompressed]

for ax, (key, col_name, text) in zip(axes.ravel(), panels):
    for r in r_pdf_marks:
        ax.axvline(r, color='grey', linestyle='-', linewidth=2.5, alpha=0.25, zorder=0)
    for k in method_keys:
        lbl, col, ls, mk = signal_styles[k]
        sub = df[df['method'] == k].sort_values('r')
        ax.plot(sub['r'], sub[col_name], color=col, linestyle=ls, marker=mk,
                markersize=3.5, alpha=0.85, label=lbl)
    ax.set_xscale('log', base=2)
    xlim = ax.get_xlim()
    ax.axvspan(hybrid_shade_left, xlim[1], color=hybrid_color, alpha=0.08, zorder=0)
    ax.set_xlim(xlim)
    ax.set_ylim(*YLIM[col_name])
    if key in 'cd':
        ax.set_xlabel(r'$r$')
    ax.grid(True, which='both', alpha=0.3)
    lx, ly = label_pos[key]
    ax.text(lx, ly, text, transform=ax.transAxes, ha='left', va='top')

handles, labels = axes[0, 0].get_legend_handles_labels()   # order: method_keys
axes[0, 0].legend(handles[2:], labels[2:], loc=legend_loc['a'], **legend_kw)
axes[0, 1].legend(handles[:2], labels[:2], loc=legend_loc['b'], **legend_kw)
fig.tight_layout()

fig_png = os.path.join(fig_dir, 'figure9.png')
fig_pdf = os.path.join(fig_dir, 'figure9.pdf')
plt.savefig(fig_png, dpi=300, bbox_inches='tight')
plt.savefig(fig_pdf, bbox_inches='tight')
plt.close()

print(f"Saved figure9.png and figure9.pdf in {fig_dir}")
print("Finished execution.")
