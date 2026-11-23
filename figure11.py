"""Figure 11 (diagnostic): E(k), S_2 and S_4 with and without the L2 renormalization of the
TT-SVD truncations.

The stored standard and hybrid TT fields were generated with tensnet's
tt.svd(renormalize=True), which after every truncation rescales the kept singular values
back to the Frobenius norm of the matrix being split. The bond-dimension truncation itself
is unchanged by this rescaling, so a stored field is exactly

    stored = c * plain,    plain = X X^T theta,

with one global scalar c >= 1 per field, where X X^T theta is the plain (non-renormalized)
TT-SVD truncation, an orthogonal projection of the DNS field theta (nested left
projections). Hence <theta, plain> = |plain|^2 and c follows from the stored field and the
DNS alone:

    c = |stored|^2 / <theta, stored>.

The hybrid seam interpolation (*_int64 fields) is linear, so int64(plain) = int64(stored)/c.
Without the renormalization, E(k) and S_2 of each snapshot scale by 1/c^2 and S_4 by 1/c^4;
scale-invariant statistics (flatness, normalized PDFs) are unchanged.

Top row: E(k), S_2(r), S_4(r). Bottom row: ratio to the DNS. Filled markers: stored fields
(renormalized, as in all paper figures); open markers: plain truncation. Galerkin and
wavelet involve no SVD and no renormalization; their curves are read from the Fig. 1 cache
(same snapshots, same compression ratio).
"""
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

import config as cfg
# spectral_tools (and the raw/compressed data) are only needed to RECOMPUTE the
# cached statistics; imported lazily inside the PLOT_ONLY=False branch below.

# ── Publication style (same as figure1.py) ─────────────────────────────────────
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
    'legend.fontsize': 6.5,
    'figure.titlesize': 10,
    'lines.linewidth': 1.0,
    'lines.markersize': 3.0,
    'axes.linewidth': 0.6,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'grid.linewidth': 0.4,
})
if _USE_TEX:
    mpl.rcParams['text.latex.preamble'] = r'\usepackage{amsmath}'

# ── Parameters ────────────────────────────────────────────────────────────────
# PLOT_ONLY=True re-plots from statistics/figure11_data.pt; False recomputes it from
# the raw DNS and the stored TT fields on the SSD (cfg.DATA_ROOT).
PLOT_ONLY = False
L = 12
BD_MAX = 100                       # rho = 1.31%, as in Fig. 1
p_list = [2, 4]
r_list = [2**i for i in range(12)]
configs_to_average = list(range(59, 109)) + list(range(509, 559))   # as in figure1.py
num_snaps = len(configs_to_average)
data_path = os.path.join(cfg.STATS_DIR, 'figure11_data.pt')
fig1_path = os.path.join(cfg.STATS_DIR, 'figure1_data.pt')

# Hand-tunable plot knobs.
FIG_WIDTH, FIG_HEIGHT = 6.5, 4.2
E_YLIM = (1e-2, None)              # a) E(k)
E_RATIO_YLIM = (0.0, 1.6)          # d) E/E_DNS
S_RATIO_YLIM = (None, None)        # e), f) S_p/S_p^DNS (None = automatic)
PANEL_LABEL_XY = [(0.04, 0.95), (0.04, 0.95), (0.04, 0.95),    # a) b) c)
                  (0.04, 0.93), (0.04, 0.93), (0.04, 0.93)]    # d) e) f)

# key: (label, color, linestyle, marker, filled marker)
curve_styles = {
    'original':                  ('DNS',                         'black',     '-',  'o', True),
    'galerkin':                  ('Galerkin',                    'green',     ':',  'v', True),
    'wavelet_db32':              ('Wavelet (db32)',              'purple',    '--', 'X', True),
    'standard_p4':               ('TT standard',                 'steelblue', '--', 's', True),
    'standard_p4_plain':         ('TT standard, not renormalized', 'steelblue', ':', 's', False),
    'hybrid_N12_p4_int64':       ('TT hybrid',                   'red',       '-.', 'D', True),
    'hybrid_N12_p4_int64_plain': ('TT hybrid, not renormalized', 'red',       ':',  'D', False),
}
tt_keys = ['standard_p4', 'hybrid_N12_p4_int64']
computed_keys = ['original'] + tt_keys + [k + '_plain' for k in tt_keys]


# ── Helpers (as in figure1.py) ────────────────────────────────────────────────
def read_config(config):
    filepath = os.path.join(cfg.DATA_DIR, f'theta.{config:05d}')
    theta32 = np.fromfile(filepath, dtype=np.float32)
    theta32 = theta32[1:-1].reshape(4096, -1)
    return theta32.view(np.float64)


def compute_increments(field, axis=1, r=1, stride=1):
    idx_start = torch.arange(0, field.shape[axis] - r, stride, device=field.device)
    idx_end = idx_start + r
    return torch.index_select(field, axis, idx_end) - torch.index_select(field, axis, idx_start)


def compute_structure_functions(field, r_list, p_list=(2, 4), axis=1):
    S = {p: [] for p in p_list}
    for r in r_list:
        inc = compute_increments(field, axis=axis, r=r).ravel()
        for p in p_list:
            S[p].append(torch.mean(torch.abs(inc)**p).item())
    return {p: torch.tensor(v, dtype=torch.float64) for p, v in S.items()}


if PLOT_ONLY:
    print(f"PLOT_ONLY=True: loading {data_path} ...")
    _d = torch.load(data_path, weights_only=False)
    k_vals, E_acc, sf_acc = _d['k_vals'], _d['E_acc'], _d['sf_acc']
    df_c = pd.DataFrame(_d['norm_factors'])
else:
    from spectral_tools import compute_isotropic_spectrum_2d
    if not os.path.isdir(cfg.DATA_ROOT):
        raise SystemExit(f"Data root {cfg.DATA_ROOT} not found: mount the SSD or set PSCALAR_DATA_ROOT.")

    sf_acc = {k: {p: torch.zeros(len(r_list), dtype=torch.float64) for p in p_list} for k in computed_keys}
    E_acc = {k: None for k in computed_keys}
    k_vals = None
    rows = []

    print(f"Recomputing for BD_MAX={BD_MAX} over {num_snaps} snapshots...")
    for config in configs_to_average:
        s = f'{config:05d}'
        theta = torch.from_numpy(read_config(config)).to(torch.float64)
        f_std = torch.load(os.path.join(cfg.STD_DIR, f'theta{s}_standard_mps_bd{BD_MAX}_p4.tens')).to(torch.float64)
        f_hyb_raw = torch.load(os.path.join(cfg.HYB_DIR, f'theta{s}_hybrid_mps_p4_N12_bd_equiv{BD_MAX}.tens')).to(torch.float64)
        f_hyb = torch.load(os.path.join(cfg.HYB_DIR, f'theta{s}_hybrid_mps_p4_N12_bd_equiv{BD_MAX}_int64.tens')).to(torch.float64)

        # Renormalization factors from the raw (non-interpolated) stored fields.
        c = {
            'standard_p4':         (torch.sum(f_std * f_std) / torch.sum(theta * f_std)).item(),
            'hybrid_N12_p4_int64': (torch.sum(f_hyb_raw * f_hyb_raw) / torch.sum(theta * f_hyb_raw)).item(),
        }
        n_theta = torch.linalg.norm(theta)
        rows.append({
            'config': config,
            'c_std': c['standard_p4'],
            'c_hyb': c['hybrid_N12_p4_int64'],
            'norm_ratio_std': (torch.linalg.norm(f_std) / n_theta).item(),      # expected 1
            'norm_ratio_hyb': (torch.linalg.norm(f_hyb_raw) / n_theta).item(),  # expected 1
        })
        print(f"  {s}: c_std = {c['standard_p4']:.5f}, c_hyb = {c['hybrid_N12_p4_int64']:.5f}")

        for key, f in (('original', theta), ('standard_p4', f_std), ('hybrid_N12_p4_int64', f_hyb)):
            sf = compute_structure_functions(f, r_list, p_list)
            kv, E = compute_isotropic_spectrum_2d(f)
            E = E.to(torch.float64)
            if k_vals is None:
                k_vals = kv
            targets = [(key, 1.0)] if key == 'original' else [(key, 1.0), (key + '_plain', c[key])]
            for tgt, cc in targets:
                for p in p_list:
                    sf_acc[tgt][p] += sf[p] / cc**p
                E_acc[tgt] = E / cc**2 if E_acc[tgt] is None else E_acc[tgt] + E / cc**2

    for k in computed_keys:
        for p in p_list:
            sf_acc[k][p] /= num_snaps
        E_acc[k] /= num_snaps

    df_c = pd.DataFrame(rows)
    os.makedirs(cfg.STATS_DIR, exist_ok=True)
    torch.save({'k_vals': k_vals, 'E_acc': E_acc, 'r_list': r_list, 'sf_acc': sf_acc,
                'norm_factors': df_c.to_dict('list'), 'BD_MAX': BD_MAX,
                'configs': configs_to_average}, data_path)
    df_c.to_csv(os.path.join(cfg.STATS_DIR, 'figure11_norm_factors.csv'), index=False)
    print(f"Saved {data_path} and figure11_norm_factors.csv")

# ── Galerkin / wavelet from the Fig. 1 cache; consistency check of the rest ──
_f1 = torch.load(fig1_path, weights_only=False)
for key in ('galerkin', 'wavelet_db32'):
    E_acc[key] = _f1['E_acc'][key].to(torch.float64)
    sf_acc[key] = {p: _f1['sf_acc'][key][p].to(torch.float64) for p in p_list}
for key in ['original'] + tt_keys:
    dE = (E_acc[key][1:] - _f1['E_acc'][key][1:].to(torch.float64)).abs().max() / E_acc[key][1:].abs().max()
    dS = max(((sf_acc[key][p] - _f1['sf_acc'][key][p].to(torch.float64)) / sf_acc[key][p]).abs().max().item()
             for p in p_list)
    print(f"Consistency with figure1_data.pt, {key}: max rel. diff E = {dE.item():.1e}, S_p = {dS:.1e}")

# ── Summary of the renormalization factors ────────────────────────────────────
for col, name in (('c_std', 'TT standard'), ('c_hyb', 'TT hybrid')):
    cvals = np.asarray(df_c[col])
    print(f"{name}: c = {cvals.mean():.5f} (min {cvals.min():.5f}, max {cvals.max():.5f}); "
          f"without renormalization E, S_2 change by {100*(np.mean(cvals**-2)-1):+.2f}%, "
          f"S_4 by {100*(np.mean(cvals**-4)-1):+.2f}%")
for col in ('norm_ratio_std', 'norm_ratio_hyb'):
    print(f"{col}: max |ratio - 1| = {np.max(np.abs(np.asarray(df_c[col]) - 1)):.1e}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(FIG_WIDTH, FIG_HEIGHT), dpi=300, sharex='col',
                         gridspec_kw={'height_ratios': [1.6, 1]})
k_np = k_vals.cpu().numpy()
kmask = k_np >= 1
E_dns = E_acc['original'].cpu().numpy()
S_dns = {p: sf_acc['original'][p].cpu().numpy() for p in p_list}
me_E = max(1, int(kmask.sum()) // 7)

for key, (lab, col, ls, mk, filled) in curve_styles.items():
    kw = dict(color=col, linestyle=ls, marker=mk, markerfacecolor=col if filled else 'none',
              markeredgewidth=0.7, alpha=0.85, label=lab)
    E = E_acc[key].cpu().numpy()
    axes[0, 0].loglog(k_np[kmask], E[kmask], markevery=me_E, **kw)
    with np.errstate(divide='ignore', invalid='ignore'):
        axes[1, 0].semilogx(k_np[kmask], np.where(E_dns[kmask] > 0, E[kmask] / E_dns[kmask], np.nan),
                            markevery=me_E, **kw)
    for j, p in enumerate(p_list, start=1):
        S = sf_acc[key][p].cpu().numpy()
        axes[0, j].loglog(r_list, S, **kw)
        axes[1, j].plot(r_list, S / S_dns[p], **kw)

for j in (1, 2):
    for i in (0, 1):
        axes[i, j].set_xscale('log', base=2)
for ax in axes[1]:
    ax.axhline(1.0, color='black', linewidth=0.8, alpha=0.6, zorder=1)
for ax in axes.flat:
    ax.grid(True, which='both', alpha=0.3)

axes[0, 0].set_ylim(*E_YLIM)
axes[1, 0].set_ylim(*E_RATIO_YLIM)
axes[1, 1].set_ylim(*S_RATIO_YLIM)
axes[1, 2].set_ylim(*S_RATIO_YLIM)
axes[1, 0].set_xlabel(r'$k$')
axes[1, 1].set_xlabel(r'$r$')
axes[1, 2].set_xlabel(r'$r$')

panel_labels = [r'a) $E(k)$', r'b) $S_2(r)$', r'c) $S_4(r)$',
                r'd) $E/E_{\rm DNS}$', r'e) $S_2/S_2^{\rm DNS}$', r'f) $S_4/S_4^{\rm DNS}$']
for ax, lab, (lx, ly) in zip(axes.flat, panel_labels, PANEL_LABEL_XY):
    ax.text(lx, ly, lab, transform=ax.transAxes, ha='left', va='top', fontsize=9)

# Legends: DNS/Galerkin/wavelet in a), the four TT curves in b).
h, l = axes[0, 0].get_legend_handles_labels()
axes[0, 0].legend(h[:3], l[:3], loc='lower left', handlelength=1.8)
axes[0, 1].legend(h[3:], l[3:], loc='lower right', handlelength=1.8)

fig.tight_layout()
os.makedirs(cfg.FIG_DIR, exist_ok=True)
plt.savefig(os.path.join(cfg.FIG_DIR, 'figure11.png'), dpi=300, bbox_inches='tight')
plt.savefig(os.path.join(cfg.FIG_DIR, 'figure11.pdf'), bbox_inches='tight')
plt.close()
print(f"Saved figure11.png and figure11.pdf in {cfg.FIG_DIR}")
