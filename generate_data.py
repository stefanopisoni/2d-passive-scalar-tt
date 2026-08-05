# %%
import sys
import os
import config as cfg   # imported as cfg because `config` is used as a loop variable below
# Add the current directory to the path so hybrid_MPS can be found
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(cfg.TENSNET_SRC)   # external tensnet library (see config.py)

# %%
from hybrid_MPS import *
import torch
import numpy as np
import matplotlib.pyplot as plt

import tensnet as tt
from spectral_tools import galerkin_truncate_2d, wavelet_truncate_2d, compute_isotropic_spectrum_2d

# %% Statistics: SF and PDF of the increments
# PDF and SF auxiliary functions
def compute_increments(field, axis=1, r=1, stride=1, inverse_stride=0):
    """
    Compute increments along a given axis with separation r.
    stride: take 1 starting point every 'stride' ones.
    inverse_stride: skip 1 starting point every 'inverse_stride' ones (e.g. at block boundaries).
    """
    idx_start = torch.arange(0, field.shape[axis]-r, stride, device=field.device)
    if inverse_stride > 0:
        # Skip indices k where (k+1) % inverse_stride == 0 (e.g. 7, 15, 23...)
        mask = (idx_start + 1) % inverse_stride != 0
        idx_start = idx_start[mask]
        
    idx_end = idx_start + r
    
    inc = torch.index_select(field, axis, idx_end) - torch.index_select(field, axis, idx_start)
    return inc

def compute_pdf(data, bins, density=True):
    """
    Compute PDF from data.
    Returns bin centers and pdf values.
    """
    hist = torch.histogram(data.cpu(), bins=bins, density=density)
    counts = hist.hist
    edges = hist.bin_edges
    centers = 0.5 * (edges[:-1] + edges[1:])
    
    return centers, counts

def compute_structure_functions(field_list, r_list, p_list=[1,2,3,4], axis=1, stride=1, normalize=True):
    S = {p: [] for p in p_list}
    for r in r_list:
        all_increments = []
        for field in field_list:
            inc = compute_increments(field, axis=axis, r=r, stride=stride).ravel()
            all_increments.append(inc)
        all_increments = torch.cat(all_increments)
        if normalize:
            all_increments = all_increments / torch.std(all_increments)
        for p in p_list:
            S[p].append(torch.mean(torch.abs(all_increments)**p).item())
    return {p: torch.tensor(vals) for p, vals in S.items()}

# %% Interleaved encoding in 2D
def shuffle_axis(A: torch.Tensor, L: int):
    new_axis = []
    for x in range(L):
        new_axis += [x, x + L]
    A = A.reshape([2] * 2 * L).permute(new_axis).reshape(2**L, 2**L)
    return A

def unshuffle_axis(A: torch.Tensor, L: int):
    axis_even_odd = [x for x in range(2 * L) if x % 2 == 0] + [
        x for x in range(2 * L) if x % 2 == 1]
    A = A.reshape([2] * (2 * L)).permute(axis_even_odd).reshape(2**L, 2**L)
    return A


# %% Load passive scalar snapshot and convert it to torch tensor
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import SymLogNorm
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import warnings
warnings.filterwarnings("ignore")

cmap = mpl.cm.get_cmap('seismic')
cmap2 = mpl.cm.get_cmap('plasma')
colors_list_4_plasma = [cmap2(0), cmap2(0.4), cmap2(0.6), cmap2(0.9)]
colors_list_4 = [cmap(0), cmap(0.4), cmap(0.6), cmap(0.9)]

colors = ['white',
         colors_list_4[1], colors_list_4[0], colors_list_4_plasma[1],
          colors_list_4[3],
          colors_list_4_plasma[3]]


cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)




#%% Read data
def read_config(config):
    # vmag32 = np.fromfile(path + 'vmag_real.{:05d}'.format(config), dtype = np.float32)
    # vmag32 = vmag32[1:-1].reshape(4096,-1)
    # vmag = vmag32.view(np.float64)
    # vmag = vmag[:,:-2]

    # vx32 = np.fromfile(path + 'vx_real.{:05d}'.format(config), dtype = np.float32)
    # vx32 = vx32[1:-1].reshape(4096,-1)
    # vx = vx32.view(np.float64)
    # vx = vx[:,:-2]

    # vy32 = np.fromfile(path + 'vy_real.{:05d}'.format(config), dtype = np.float32)
    # vy32 = vy32[1:-1].reshape(4096,-1)
    # vy = vy32.view(np.float64)
    # vy = vy[:,:-2]

    # omega32 = np.fromfile(path + 'omega.{:05d}'.format(config), dtype = np.float32)
    # omega32 = omega32[1:-1].reshape(4096,-1)
    # omega = omega32.view(np.float64)

    theta32 = np.fromfile(path + 'theta.{:05d}'.format(config), dtype = np.float32)
    theta32 = theta32[1:-1].reshape(4096,-1)
    theta = theta32.view(np.float64)

    # return vmag, vx, vy, theta, omega
    # return theta, omega
    return theta

# vmag, vx, vy, theta, omega = read_config(config)
# theta, omega = read_config(config)

# Conversion to torch to handle tensnet native format
# vmag = torch.from_numpy(vmag)
# vx = torch.from_numpy(vx)
# vy = torch.from_numpy(vy)
# omega = torch.from_numpy(omega)
# theta = torch.from_numpy(theta)

# print(theta.dtype)
# plt.imshow(theta, vmin = -2, vmax = 2,  cmap=cmap)
# plt.colorbar()
# plt.show()


# %% Convert to hybrid_mps and standard_mps with proper dof counting
L = 12
path = cfg.DATA_DIR
# config = 59

configs_to_process = list(range(59, 69)) + list(range(549, 559))
for config in configs_to_process:

    theta, omega = read_config(config)
    theta = torch.from_numpy(theta)
    # omega = torch.from_numpy(omega)
    print(config)

    for BD_MAX in [100, 200, 500, 1000]:

        names = ["theta"]
        for name, snap in zip(names, [theta]):

            vec = shuffle_axis(snap, L).flatten()

            # standard MPS
            standard_mps = tt.vector_to_mps(vec, 2, 2*L, bdmax=BD_MAX)
            print(standard_mps)
            torch.save(standard_mps.compression, f'{cfg.STD_DIR}/{name}{config:05d}_standard_mps_bd{BD_MAX}.compression')
            vec_new_standard = standard_mps.contract()
            vec_new_standard = unshuffle_axis(vec_new_standard.reshape((2**L,2**L)), L)
            torch.save(vec_new_standard, f'{cfg.STD_DIR}/{name}{config:05d}_standard_mps_bd{BD_MAX}.tens')

            # hybrid MPS
            # for N in [8, 12, 16]:
            for N in [12]:

                # Calculate effective BD for the hybrid MPS
                dof_standard_mps = standard_mps.num_params
                print(dof_standard_mps)
                if dof_standard_mps - (2 * 2**N) < 0:
                    print("BD_MAX is too small for the chosen N.")
                    break

                # must depend on N as well
                if BD_MAX == 100:
                    if N == 8:
                        BD_REMAINING = 103
                    elif N == 12:
                        BD_REMAINING = 49
                    elif N == 16:
                        BD_REMAINING = 3

                elif BD_MAX == 200:
                    if N == 8:
                        BD_REMAINING = 202
                    elif N == 12:
                        BD_REMAINING = 138
                    elif N == 16:
                        BD_REMAINING = 12

                elif BD_MAX == 500:
                    if N == 8:
                        BD_REMAINING = 495
                    elif N == 12:
                        BD_REMAINING = 478
                    elif N == 16:
                        BD_REMAINING = 55

                elif BD_MAX == 1000:
                    if N == 8:
                        BD_REMAINING = 1001
                    elif N == 12:
                        BD_REMAINING = 1116
                    elif N == 16:
                        BD_REMAINING = 163


                # BD_REMAINING = sqrt((dof_standard_mps - (BD_REMAINING *2**N)) / (2*(2*L-N))), second order.
                # BD_REMAINING = int(((2**(2*N) + 8*dof_standard_mps*(2*L-N))**0.5 - 2**N) / (4*(2*L-N)))
                print(BD_REMAINING)

                hybrid_mps = vector_to_hybrid_mps(vec, 2*L, N, bd_first = BD_REMAINING, bd_remaining = BD_REMAINING)
                print(hybrid_mps)
                torch.save(hybrid_mps.compression, f'{cfg.HYB_DIR}/{name}{config:05d}_hybrid_mps_N{N}_bd_equiv{BD_MAX}.compression')
                vec_new_hybrid = hybrid_mps.contract()
                vec_new_hybrid = unshuffle_axis(vec_new_hybrid.reshape((2**L,2**L)), L)
                torch.save(vec_new_hybrid, f'{cfg.HYB_DIR}/{name}{config:05d}_hybrid_mps_N{N}_bd_equiv{BD_MAX}.tens')

# %% Convert to mps using p=4
L = 12
path = cfg.DATA_DIR
r_PDF = 1

# configs_to_process= [59]
configs_to_process = list(range(59, 109)) + list(range(509, 559))
for config in configs_to_process:

    theta = read_config(config)
    theta = torch.from_numpy(theta)
    print(config)

    for BD_MAX in [1000]:
        names = ["theta"]
        for name, snap in zip(names, [theta]):

            vec = shuffle_axis(snap, L).flatten()

            # standard MPS
            standard_mps = tt.vector_to_mps(vec, 4, L, bdmax=BD_MAX)
            print(standard_mps)
            torch.save(standard_mps.compression, f'{cfg.STD_DIR}/{name}{config:05d}_standard_mps_bd{BD_MAX}_p4.compression')
            vec_new_standard = standard_mps.contract()
            vec_new_standard = unshuffle_axis(vec_new_standard.reshape((2**L,2**L)), L)
            torch.save(vec_new_standard, f'{cfg.STD_DIR}/{name}{config:05d}_standard_mps_bd{BD_MAX}_p4.tens')


            # # --- Comparison Plot: p=2 vs p=4 ---
            # f_p4 = vec_new_standard
            # f_p2 = torch.load(f'{cfg.STD_DIR}/{name}{config:05d}_standard_mps_bd{BD_MAX}.tens')
            # f_orig = snap
            
            # # Compute stats
            # r_list_plot = [2**i for i in range(12)]
            # p_plot = [2, 4, 6]
            # bins_plot = torch.arange(-20, 20, 0.1)
            
            # sf_orig = compute_structure_functions([f_orig], r_list_plot, p_list=p_plot, axis=1, stride=1, normalize=False)
            # sf_p2 = compute_structure_functions([f_p2], r_list_plot, p_list=p_plot, axis=1, stride=1, normalize=False)
            # sf_p4 = compute_structure_functions([f_p4], r_list_plot, p_list=p_plot, axis=1, stride=1, normalize=False)
            
            # # Spectra
            # k_orig, E_orig = compute_isotropic_spectrum_2d(f_orig)
            # k_p2, E_p2 = compute_isotropic_spectrum_2d(f_p2)
            # k_p4, E_p4 = compute_isotropic_spectrum_2d(f_p4)
            
            # # PDFs
            # def get_pdf(f):
            #     all_dt = compute_increments(f, axis=1, r=r_PDF, stride=1).ravel()
            #     all_dt_norm = all_dt / torch.std(all_dt)
            #     _, counts = compute_pdf(all_dt_norm, bins=bins_plot.to(dtype=torch.float64))
            #     return counts
            # pdf_orig = get_pdf(f_orig)
            # pdf_p2 = get_pdf(f_p2)
            # pdf_p4 = get_pdf(f_p4)
            
            # # Plot
            # fig, axes = plt.subplots(1, 3, figsize=(20, 6))
            # # Flatness
            # axes[0].loglog(r_list_plot, sf_orig[4] / sf_orig[2]**2, 'k-', label='Original')
            # axes[0].loglog(r_list_plot, sf_p2[4] / sf_p2[2]**2, 'r--', label='p=2')
            # axes[0].loglog(r_list_plot, sf_p4[4] / sf_p4[2]**2, 'b:', label='p=4')
            # axes[0].set_title('Flatness: $S_4 / S_2^2$')
            # axes[0].grid(True, which='both', alpha=0.3)
            # axes[0].legend()
            
            # # PDF
            # axes[1].semilogy(bins_plot[:-1], pdf_orig, 'k-', label='Original')
            # axes[1].semilogy(bins_plot[:-1], pdf_p2, 'r--', label='p=2')
            # axes[1].semilogy(bins_plot[:-1], pdf_p4, 'b:', label='p=4')
            # axes[1].set_title(f'PDF (r={r_PDF})')
            # axes[1].grid(True, which='both', alpha=0.3)
            # axes[1].legend()
            
            # # Spectrum
            # axes[2].loglog(k_orig, E_orig, 'k-', label='Original')
            # axes[2].loglog(k_p2, E_p2, 'r--', label='p=2')
            # axes[2].loglog(k_p4, E_p4, 'b:', label='p=4')
            # axes[2].set_title('Energy Spectrum')
            # axes[2].set_ylim(1e-2, None)
            # axes[2].grid(True, which='both', alpha=0.3)
            # axes[2].legend()
            
            # plt.suptitle(f'Comparison p=2 vs p=4 (BD=100, config={config})', fontsize=16, fontweight='bold')
            # plt.tight_layout()
            # plt.show()

# %% Convert to hybrid with p=4

# Empirical counting of dof to estimate the best BD_REM.
L = 12
path = cfg.DATA_DIR

configs_to_process = list(range(59,69))
# configs_to_process = list(range(59, 109)) + list(range(509, 559))
for config in configs_to_process:

    theta = read_config(config)
    theta = torch.from_numpy(theta)
    print(config)

    for BD_MAX in [50,75,100,200,500,1000]:

        names = ["theta"]
        for name, snap in zip(names, [theta]):

            vec = shuffle_axis(snap, L).flatten()

            # hybrid MPS with p=4
            for N in [8,16]:

                if BD_MAX == 50:
                    if N == 8:
                        BD_REMAINING = 50
                    if N == 12:
                        BD_REMAINING = 16
                    if N == 16:
                        BD_REMAINING = 1
                    
                if BD_MAX == 75:
                    if N == 8:
                        BD_REMAINING = 76
                    if N == 12:
                        BD_REMAINING = 31
                    if N == 16:
                        BD_REMAINING = 2

                elif BD_MAX == 100:
                    if N == 8:
                        BD_REMAINING = 101
                    if N == 12:
                        BD_REMAINING = 49
                    if N == 16:
                        BD_REMAINING = 3    # -10%, BD=4 wou be +20%

                elif BD_MAX == 200:
                    if N == 8:
                        BD_REMAINING = 201
                    if N == 12:
                        BD_REMAINING = 138
                    if N == 16:
                        BD_REMAINING = 11 # -4%

                elif BD_MAX == 500:
                    if N == 8:
                        BD_REMAINING = 458
                    if N == 12:
                        BD_REMAINING = 448
                    if N == 16:
                        BD_REMAINING = 48

                elif BD_MAX == 1000:
                    if N == 8:
                        BD_REMAINING = 867
                    if N == 12:
                        BD_REMAINING = 1107
                    if N == 16:
                        BD_REMAINING = 155

                print(BD_REMAINING)

                hybrid_mps_p4 = vector_to_hybrid_mps_p4(vec, 2*L, N, bd_first=BD_REMAINING, bd_remaining=BD_REMAINING)
                print(hybrid_mps_p4)
                torch.save(hybrid_mps_p4.compression, f'{cfg.HYB_DIR}/{name}{config:05d}_hybrid_mps_p4_N{N}_bd_equiv{BD_MAX}.compression')
                vec_new_hybrid_p4 = hybrid_mps_p4.contract()
                vec_new_hybrid_p4 = unshuffle_axis(vec_new_hybrid_p4.reshape((2**L, 2**L)), L)
                torch.save(vec_new_hybrid_p4, f'{cfg.HYB_DIR}/{name}{config:05d}_hybrid_mps_p4_N{N}_bd_equiv{BD_MAX}.tens')


# %% Generate hybrid_Np4_int64: replace every coarse-block-boundary column with the
# linear x-interpolation of its two neighbours. The seams are the coarse-block
# boundaries: with the interleaved encoding the first tensor holds the N/2 MSBs of
# each axis, so fine-x has L-N/2 bits and a coarse block is 2^(L-N/2) columns wide.
# The flagged columns are therefore where (x+1) % 2^(L-N/2) == 0. This equals 64
# only at N=12 (=L); for other N the period differs (e.g. 256 for N=8, 16 for N=16).
# Periodic wrapping for the last column. NOTE: earlier this used 2^(N/2), which is
# the reciprocal period and only correct at N=12.
field_size = 4096

for N in [8,16]:

    seam_period = field_size // (2**(N // 2))                            # 2^(L - N/2)
    interp_cols = np.where((np.arange(field_size) + 1) % seam_period == 0)[0]

    configs_to_process_int64 = list(range(59,69))
    # configs_to_process_int64 = list(range(59, 109)) + list(range(509, 559))
    for BD_MAX in [50,75,100,200,500,1000]:
        for config in configs_to_process_int64:
            snap_str = f"{config:05d}"
            src = f'{cfg.HYB_DIR}/theta{snap_str}_hybrid_mps_p4_N{N}_bd_equiv{BD_MAX}.tens'
            field = torch.load(src).numpy()

            field_int64 = field.copy()
            for x_col in interp_cols:
                x_left  = x_col - 1                    # always valid (x_col >= 63)
                x_right = (x_col + 1) % field_size     # periodic: 4096 -> 0
                field_int64[:, x_col] = 0.5 * (field[:, x_left] + field[:, x_right])

            dst = f'{cfg.HYB_DIR}/theta{snap_str}_hybrid_mps_p4_N{N}_bd_equiv{BD_MAX}_int64.tens'
            torch.save(torch.from_numpy(field_int64), dst)
            print(f'Saved int64-interpolated field for config {config}, N={N}')


# %% Generate hybrid_N12_p4_int64_xy: local 1-D interpolation of both seams.
# The y-interpolation mirrors the x-interpolation (neighbouring points): flagged columns x with (x+1)%64==0 are replaced by 0.5*(field[:,x-1] +
# field[:,x+1]) (E-W), and flagged rows y with (y+1)%64==0 by 0.5*(field[y-1,:] +
# field[y+1,:]) (N-S). Cells on both a flagged row and column (the 64x64 seam
# crossings) take the average of the two 1-D interpolations. Periodic wrapping.
# Self-contained: the seam period is derived from N=12 (2^(N/2) = 64).
field_size = 4096
seam_period = 2 ** (12 // 2)                                                # 64
interp_idx = np.where((np.arange(field_size) + 1) % seam_period == 0)[0]    # 63,127,...,4095
col_flag = np.zeros(field_size, dtype=bool); col_flag[interp_idx] = True
row_flag = np.zeros(field_size, dtype=bool); row_flag[interp_idx] = True

configs_to_process_int64_xy = list(range(59, 109)) + list(range(509, 559))
BD_MAX = 100
for config in configs_to_process_int64_xy:
    snap_str = f"{config:05d}"
    src = f'{cfg.HYB_DIR}/theta{snap_str}_hybrid_mps_p4_N12_bd_equiv{BD_MAX}.tens'
    field = torch.load(src).numpy()

    # 1-D interpolants of every cell (periodic neighbours from the original field).
    x_interp = 0.5 * (np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1))  # 0.5*(x-1, x+1)
    y_interp = 0.5 * (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0))  # 0.5*(y-1, y+1)

    C = col_flag[None, :]        # flagged columns (broadcast over rows)
    R = row_flag[:, None]        # flagged rows    (broadcast over columns)
    only_col = C & ~R
    only_row = R & ~C
    corner   = C & R

    field_int64_xy = field.copy()
    field_int64_xy[only_col] = x_interp[only_col]
    field_int64_xy[only_row] = y_interp[only_row]
    field_int64_xy[corner]   = 0.5 * (x_interp[corner] + y_interp[corner])

    dst = f'{cfg.HYB_DIR}/theta{snap_str}_hybrid_mps_p4_N12_bd_equiv{BD_MAX}_int64_xy.tens'
    torch.save(torch.from_numpy(field_int64_xy), dst)
    print(f'Saved int64-xy-interpolated field for config {config}')


# %% Check: compare num_params across encodings for a given config and BD_MAX
# Compression ratio is stored as a scalar; num_params = compression * 2^(2*L) = compression * 2^24
check_config = 59
check_L = 12
total_size = 2 ** (2 * check_L)  # 4096 * 4096 = 2^24

print(f"{'BD_MAX':>8}  {'Encoding':<28}  {'compression':>12}  {'num_params':>12}")
print("-" * 70)
for BD_MAX in [100, 200, 500, 1000]:
    snap_str = f"{check_config:05d}"

    comp_std_p2 = torch.load(
        f'{cfg.STD_DIR}/theta{snap_str}_standard_mps_bd{BD_MAX}.compression'
    )
    comp_std_p4 = torch.load(
        f'{cfg.STD_DIR}/theta{snap_str}_standard_mps_bd{BD_MAX}_p4.compression'
    )
    comp_hyb_p2 = torch.load(
        f'{cfg.HYB_DIR}/theta{snap_str}_hybrid_mps_N12_bd_equiv{BD_MAX}.compression'
    )
    comp_hyb_p4 = torch.load(
        f'{cfg.HYB_DIR}/theta{snap_str}_hybrid_mps_p4_N12_bd_equiv{BD_MAX}.compression'
    )

    for label, comp in [
        ('standard p=2',    comp_std_p2),
        ('standard p=4',    comp_std_p4),
        ('hybrid N=12 p=2', comp_hyb_p2),
        ('hybrid N=12 p=4', comp_hyb_p4),
    ]:
        num_params = int(comp * total_size)
        print(f"{BD_MAX:>8}  {label:<28}  {comp:>12.6f}  {num_params:>12d}")
    print()















# Checks not needed to produce the figures in the paper.
# Focus on 1D stats to understand the compression effects.

# can be entirely skipped for paper purposes.

# %% Statistics: SF and PDF of the increments
L = 12
path = cfg.DATA_DIR

# ── Parameters ────────────────────────────────────────────────────────────────
downsample = 1
BD_MAX     = 100
p_list     = [2, 3, 4, 5, 6]
r_list     = [2**i for i in range(12)]
r_pdf_list = [1, 2, 4, 8, 16]          # scales at which PDFs are computed
bins       = torch.arange(-20, 20, 0.1)
configs_to_average = list(range(59, 69)) 
# + list(range(549, 559))

ratios = [
    ('Flatness: $S_4/S_2^2$',     4, 2, 2,   (1, 100)),
    ('$S_6/S_3^2$',               6, 3, 2,   (1, 10000)),
    ('Skewness: $S_3/S_2^{3/2}$', 3, 2, 1.5, (1, 10)),
]

# ── Signal catalogue ──────────────────────────────────────────────────────────
# (label, line color, line style, marker)
signal_styles = {
    'original':            ('Original',              'black',    '-',   'o'),
    'standard_p4':         ('Standard $p=4$',        'steelblue',      '--',  's'),
    'hybrid_N12_p4':       ('Hybrid $p=4$ $N=12$',   'royalblue','-',   '^'),
    'hybrid_N12_p4_int64': ('Hybrid $p=4$ $N=12$ int64',    'red','-.', 'D'),
    'galerkin':            ('Galerkin',              'green',    ':',   'v'),
    'wavelet':             ('Wavelet',               'darkorange','--',  'P'),
}
signal_keys = list(signal_styles.keys())

# ── Accumulators ──────────────────────────────────────────────────────────────
sf_acc    = {k: {p: torch.zeros(len(r_list))   for p in p_list}  for k in signal_keys}
sf_sq_acc = {k: {p: torch.zeros(len(r_list))   for p in p_list}  for k in signal_keys}
pdf_acc = {k: {r: torch.zeros(len(bins)-1)   for r in r_pdf_list} for k in signal_keys}
E_acc   = {k: None for k in signal_keys}
k_vals  = None
cr_acc  = 0.0
num_snaps = len(configs_to_average)

def get_pdf_at_r(f, r):
    inc  = compute_increments(f, axis=1, r=r, stride=downsample).ravel()
    _, c = compute_pdf(inc / torch.std(inc), bins=bins.to(dtype=torch.float64))
    return c

# ── Main averaging loop ────────────────────────────────────────────────────────
for config in configs_to_average:
    snap_str = f"{config:05d}"
    print(f"Processing config {config}...")

    # Load original
    theta_raw = read_config(config)
    f_orig = torch.from_numpy(theta_raw)

    # DOF target from standard p=2 compression ratio (for Galerkin/wavelet matching)
    std_p2_cr  = torch.load(f'{cfg.STD_DIR}/theta{snap_str}_standard_mps_bd{BD_MAX}.compression')
    cr_acc    += std_p2_cr
    dof_target = std_p2_cr * (2**24)

    # Build field dict for this snapshot
    f_galerkin, _, _ = galerkin_truncate_2d(f_orig, dof_target)
    f_wavelet        = wavelet_truncate_2d(f_orig, dof_target)

    fields = {
        'original':            f_orig,
        'standard_p4':         torch.load(f'{cfg.STD_DIR}/theta{snap_str}_standard_mps_bd{BD_MAX}_p4.compression'.replace('.compression', '.tens')),
        'hybrid_N12_p4':       torch.load(f'{cfg.HYB_DIR}/theta{snap_str}_hybrid_mps_p4_N12_bd_equiv{BD_MAX}.tens'),
        'hybrid_N12_p4_int64': torch.load(f'{cfg.HYB_DIR}/theta{snap_str}_hybrid_mps_p4_N12_bd_equiv{BD_MAX}_int64.tens'),
        'galerkin':            f_galerkin,
        'wavelet':             f_wavelet,
    }

    for k, f in fields.items():
        # Structure functions
        sf = compute_structure_functions([f], r_list, p_list=p_list,
                                         axis=1, stride=downsample, normalize=False)
        for p in p_list:
            sf_acc[k][p]    += sf[p]
            sf_sq_acc[k][p] += sf[p] ** 2

        # PDFs at each scale
        for r in r_pdf_list:
            pdf_acc[k][r] += get_pdf_at_r(f, r)

        # Energy spectrum
        curr_k, E = compute_isotropic_spectrum_2d(f)
        if k_vals is None:
            k_vals = curr_k
        if E_acc[k] is None:
            E_acc[k] = torch.zeros_like(E)
        E_acc[k] += E

# ── Finalize averages ──────────────────────────────────────────────────────────
for k in signal_keys:
    for p in p_list:
        sf_acc[k][p]    /= num_snaps
        sf_sq_acc[k][p] /= num_snaps
    for r in r_pdf_list:
        pdf_acc[k][r] /= num_snaps
    E_acc[k] /= num_snaps
avg_cr = cr_acc / num_snaps
# std across snapshots: sqrt(E[x^2] - E[x]^2), clamped to 0 for single-snap case
sf_std = {k: {p: torch.sqrt(torch.clamp(sf_sq_acc[k][p] - sf_acc[k][p]**2, min=0))
              for p in p_list}
          for k in signal_keys}
print(f"Done. Avg compression ratio (std p=2, BD={BD_MAX}): {avg_cr:.4f}")

# ── Plot helpers ───────────────────────────────────────────────────────────────
mk_kw    = dict(markersize=9, markevery=2)   # markers for SF plots
title_kw = dict(fontsize=13, fontweight='bold')
leg_kw   = dict(fontsize=10, loc='best')
suptitle = f'BD={BD_MAX}, stride={downsample}, {num_snaps} snap(s), avg CR={avg_cr:.4f}'

import matplotlib.colors as mcolors

def _lighten(color, amount=0.5):
    """Return a lighter version of `color` by blending with white."""
    try:
        c = mcolors.to_rgb(color)
    except ValueError:
        c = mcolors.to_rgb(mcolors.CSS4_COLORS.get(color, color))
    return tuple(1 - amount * (1 - ch) for ch in c)

def plot_signals_loglog(ax, x_vals, data_fn, xlabel='r', ylabel=None, std_fn=None):
    x_arr = np.array(x_vals, dtype=float)
    for k in signal_keys:
        lbl, col, ls, mk = signal_styles[k]
        y = np.asarray(data_fn(k), dtype=float)
        ax.loglog(x_arr, y, color=col, linestyle=ls, marker=mk,
                  label=lbl, alpha=0.85, linewidth=1.6, **mk_kw)
        if std_fn is not None:
            sigma = np.asarray(std_fn(k), dtype=float)
            lo = np.maximum(y - sigma, 1e-30)
            ax.fill_between(x_arr, lo, y + sigma,
                            color=_lighten(col, 0.45), alpha=0.30, linewidth=0)
    ax.set_xlabel(xlabel, fontsize=11)
    if ylabel: ax.set_ylabel(ylabel, fontsize=11)
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(**leg_kw)

# ── Figure 1: Structure functions S2–S6 (2 rows × 3 cols) ─────────────────────
fig1, axes1 = plt.subplots(2, 3, figsize=(21, 12))
for idx, p in enumerate(p_list):
    ax = axes1[idx // 3, idx % 3]
    plot_signals_loglog(ax, r_list,
                        lambda k, p=p: sf_acc[k][p].numpy(),
                        std_fn=lambda k, p=p: sf_std[k][p].numpy(),
                        xlabel='r (grid spacing)')
    ax.set_title(f'$S_{p}(r)$', **title_kw)
axes1[1, 2].set_visible(False)   # 6th slot unused
plt.suptitle(f'Structure Functions — {suptitle}', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ── Figure 2: Ratios — Flatness, S6/S3^2, Skewness (1 row × 3 cols) ──────────
fig2, axes2 = plt.subplots(1, 3, figsize=(21, 6))
for ax, (title, num, den, pwr, ylim) in zip(axes2, ratios):
    plot_signals_loglog(ax, r_list,
                        lambda k, num=num, den=den, pwr=pwr:
                            (sf_acc[k][num] / sf_acc[k][den]**pwr).numpy(),
                        xlabel='r (grid spacing)')
    ax.set_title(title, **title_kw)
    ax.set_ylim(ylim)
    ax.legend(**leg_kw)
plt.suptitle(f'Moment Ratios — {suptitle}', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ── Figure 3: Energy spectrum ──────────────────────────────────────────────────
fig3, ax3 = plt.subplots(1, 1, figsize=(9, 7))
n_k = len(k_vals)
spec_mk_kw = dict(markersize=6, markevery=max(1, n_k // 10))   # ~10 markers per curve
for k in signal_keys:
    lbl, col, ls, mk = signal_styles[k]
    ax3.loglog(k_vals, E_acc[k].numpy(), color=col, linestyle=ls, marker=mk,
               label=lbl, alpha=0.85, linewidth=1.4, **spec_mk_kw)
k_ref = k_vals[10:100]
E_ref = E_acc['original'][10] * (k_ref / k_ref[0])**(-5/3)
ax3.loglog(k_ref, E_ref.numpy(), 'k--', linewidth=1.8, label=r'$k^{-5/3}$')
ax3.set_xlabel('k (wavenumber)', fontsize=12)
ax3.set_ylabel('E(k)', fontsize=12)
ax3.set_ylim(1e-2, None)
ax3.set_title('Isotropic Energy Spectrum', **title_kw)
ax3.grid(True, which='both', alpha=0.35)
ax3.legend(fontsize=10)
plt.suptitle(f'Energy Spectrum — {suptitle}', fontsize=13, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

# ── Figure 4: PDFs — row 1: r=1,2 | row 2: r=4,8,16 (2 rows × 3 cols) ────────
fig4, axes4 = plt.subplots(2, 3, figsize=(21, 12))
r_layout = [(0,0,1), (0,1,2), (1,0,4), (1,1,8), (1,2,16)]
for row, col, r in r_layout:
    ax = axes4[row, col]
    for k in signal_keys:
        lbl, color, ls, mk = signal_styles[k]
        ax.semilogy(bins[:-1].numpy(), pdf_acc[k][r].numpy(),
                    color=color, linestyle=ls,
                    label=lbl, alpha=0.85, linewidth=1.4)
    ax.set_title(f'PDF, $r={r}$', **title_kw)
    ax.set_xlabel('Normalised increment $\\delta\\theta / \\sigma$', fontsize=11)
    ax.set_ylabel('Probability density', fontsize=11)
    ax.set_xlim(-20, 20)
    ax.grid(True, which='both', alpha=0.35)
    ax.legend(fontsize=10)
axes4[0, 2].set_visible(False)   # top-right slot unused (only 2 PDFs in row 1)
plt.suptitle(f'Increment PDFs — {suptitle}', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()














#%% Check 1D snap signals to identify extreme events

path = cfg.DATA_DIR

# --- Parameters (edit as needed) ---
slice_config = 59          # which snapshot to load
slice_BD_MAX = 100         # bond dimension for the compressed field
slice_y_start = 3000         # start row index for the 1D slice range
slice_y_end = 3001         # end row index (exclusive)
slice_y_list = list(range(slice_y_start, slice_y_end))
zoom_x_min, zoom_x_max = 2810, 2820  # x-interval for zoomed plots
downsample = 1
inverse_stride = 0         # if > 0, skip 1 point every N ones (inverse downsampling)

# Choose compressed methods to compare
slice_method_1 = 'hybrid_N12_p4'
slice_method_2 = 'hybrid_N12_p4_int64'
increment_r = 1            # separation for the increment u(x+r) - u(x)
filter_per_point = True   # True: filter individual points (x,y); False: filter based on ensemble-mean profile
plot_pdf_rel_error = True  # True: plot error PDFs in Col 3; False: plot spatial profiles
fig2_log_y = True        # True: plot Figure 2 (Moments) in log-y scale
plot_fig1 = True          # True: plot both figures; False: plot ONLY Figure 2
r_values_fig3 = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048] # scales for Figure 3
flatness_threshold_multiplier = 0.3 # multiplier for automatic threshold (e.g. 1.0 or 2.0)
incr_tresh = 1e-2 # threshold for reporting large increments in Fig 2 reporting
inf_incr_thresh = 2e-4 # threshold for reporting extremely small increments in Fig 2 reporting
deriv_order = 1   # finite-difference stencil order: 2 (central) or 4 (higher-order central)

snapshot_str = f"{slice_config:05d}"
base_dir = cfg.DATA_ROOT

# --- Helper to load compressed fields ---
def load_compressed_field(method, config, BD, base_dir, f_orig):
    snapshot_str = f"{config:05d}"
    if method == 'standard_p2':
        path = f'{base_dir}standard_mps_data/theta{snapshot_str}_standard_mps_bd{BD}.tens'
        label = f'Standard p=2 (BD={BD})'
        return torch.load(path), label
    elif method == 'standard_p4':
        path = f'{base_dir}standard_mps_data/theta{snapshot_str}_standard_mps_bd{BD}_p4.tens'
        label = f'Standard p=4 (BD={BD})'
        return torch.load(path), label
    elif method == 'hybrid_N8':
        path = f'{base_dir}hybrid_mps_data/theta{snapshot_str}_hybrid_mps_N8_bd_equiv{BD}.tens'
        label = f'Hybrid N=8 (BD={BD})'
        return torch.load(path), label
    elif method == 'hybrid_N12':
        path = f'{base_dir}hybrid_mps_data/theta{snapshot_str}_hybrid_mps_N12_bd_equiv{BD}.tens'
        label = f'Hybrid N=12 (BD={BD})'
        return torch.load(path), label
    elif method == 'hybrid_N16':
        path = f'{base_dir}hybrid_mps_data/theta{snapshot_str}_hybrid_mps_N16_bd_equiv{BD}.tens'
        label = f'Hybrid N=16 (BD={BD})'
        return torch.load(path), label
    elif method == 'hybrid_N12_p4':
        path = f'{base_dir}hybrid_mps_data/theta{snapshot_str}_hybrid_mps_p4_N12_bd_equiv{BD}.tens'
        label = f'Hybrid p=4 N=12 (BD={BD})'
        return torch.load(path), label
    elif method == 'hybrid_N12_p4_int64':
        path = f'{base_dir}hybrid_mps_data/theta{snapshot_str}_hybrid_mps_p4_N12_bd_equiv{BD}_int64.tens'
        label = f'Hybrid p=4 N=12 int64 (BD={BD})'
        return torch.load(path), label
    elif method == 'hybrid_N12_p4_int64_xy':
        path = f'{base_dir}hybrid_mps_data/theta{snapshot_str}_hybrid_mps_p4_N12_bd_equiv{BD}_int64_xy.tens'
        label = f'Hybrid p=4 N=12 int64-xy (BD={BD})'
        return torch.load(path), label
    elif method in ('galerkin', 'wavelet'):
        std_cr = torch.load(f'{base_dir}standard_mps_data/theta{snapshot_str}_standard_mps_bd{BD}.compression')
        dof_target = std_cr * (2**24)
        if method == 'galerkin':
            field, _, _ = galerkin_truncate_2d(f_orig, dof_target)
            return field, f'Galerkin (Matched BD={BD})'
        else:
            field = wavelet_truncate_2d(f_orig, dof_target)
            return field, f'Wavelet (Matched BD={BD})'
    else:
        raise ValueError(f"Unknown method: {method}")

# --- Load Original and Compressed fields ---
theta_raw = read_config(slice_config)
f_slice_orig = torch.from_numpy(theta_raw)
orig_slices = f_slice_orig[slice_y_list, :].numpy()

methods = [slice_method_1, slice_method_2]
comp_data = {}

for m in methods:
    field, label = load_compressed_field(m, slice_config, slice_BD_MAX, base_dir, f_slice_orig)
    slices_torch = field[slice_y_list, :]
    comp_data[m] = {'slices': slices_torch.numpy(), 'slices_torch': slices_torch, 'label': label}

# --- 1D Processing Loop ---
# Original stats
orig_mean = np.mean(orig_slices, axis=0)
orig_std = np.std(orig_slices, axis=0)
orig_global_mean = np.mean(orig_mean)
x = np.arange(f_slice_orig.shape[1])

# Derivatives
def deriv_4th(f):
    return (-np.roll(f, -2, axis=-1) + 8*np.roll(f, -1, axis=-1) - 8*np.roll(f, 1, axis=-1) + np.roll(f, 2, axis=-1)) / 12.0

def deriv_2nd(f):
    return (np.roll(f, -1, axis=-1) - np.roll(f, 1, axis=-1)) / 2.0

def deriv_1st(f):
    return (np.roll(f, -1, axis=-1) - f) / 1.0

def deriv(f):
    if deriv_order == 2:
        return deriv_2nd(f)
    elif deriv_order == 4:
        return deriv_4th(f)
    elif deriv_order == 1:
        return deriv_1st(f)
    else:
        raise ValueError(f"Unknown derivative order: {deriv_order}")

dorig_slices = deriv(orig_slices)
dorig_mean = np.mean(dorig_slices, axis=0)
dorig_std = np.std(dorig_slices, axis=0)

# Increments
delta_orig_torch = compute_increments(f_slice_orig[slice_y_list, :], axis=1, r=increment_r, stride=downsample, inverse_stride=inverse_stride)
delta_orig_slices = delta_orig_torch.cpu().numpy()
delta_orig_mean = np.mean(delta_orig_slices, axis=0)
delta_orig_sq_mean = np.mean(delta_orig_slices**2, axis=0)
delta_orig_qu_mean = np.mean(delta_orig_slices**4, axis=0)
S2_orig_avg = np.mean(delta_orig_sq_mean)
S4_orig_avg = np.mean(delta_orig_qu_mean)
S2sq_orig_avg = S2_orig_avg**2

# x coordinate for increment-based plots
x_inc = np.arange(0, f_slice_orig.shape[1] - increment_r, downsample)
if inverse_stride > 0:
    x_inc = x_inc[(x_inc + 1) % inverse_stride != 0]

# Flatness and Conditional Flatness
flatness_orig_signal = delta_orig_qu_mean / S2sq_orig_avg
flatness_orig_avg = S4_orig_avg / S2sq_orig_avg

# Compute 2D Point Flatness map for original signal to set the threshold
flatness_2d_orig = delta_orig_slices**4 / S2sq_orig_avg

# Set threshold automatically to double of the maximum point-wise flatness in original data
max_val_orig = np.max(flatness_2d_orig)
max_idx_orig = np.unravel_index(np.argmax(flatness_2d_orig), flatness_2d_orig.shape)
max_y_coord = slice_y_list[max_idx_orig[0]]
max_x_coord = max_idx_orig[1]

flatness_threshold = flatness_threshold_multiplier * max_val_orig
print(f"Max point-wise original F: {max_val_orig:.2f} at (x, y) = ({max_x_coord}, {max_y_coord})")
print(f"Automatically set flatness_threshold = {flatness_threshold:.2f} ({flatness_threshold_multiplier} * max point-wise original F)")

mask_orig = np.ones_like(flatness_2d_orig if filter_per_point else flatness_orig_signal, dtype=bool)
n_remaining_orig = mask_orig.sum()
n_total_orig = mask_orig.size
flatness_orig_filtered_avg = np.mean(flatness_2d_orig if filter_per_point else flatness_orig_signal)
removed_x_orig = []
perc_removed_orig = 0.0

for m in methods:
    slices = comp_data[m]['slices']
    
    # Field Stats
    c_mean = np.mean(slices, axis=0)
    c_std = np.std(slices, axis=0)
    c_global_mean = np.mean(c_mean)
    
    # Differences
    diff_slices = np.abs(orig_slices - slices)
    rel_diff_slices = np.where(np.abs(orig_slices) > 0, diff_slices / np.abs(orig_slices), diff_slices)
    rel_diff_mean_prof = np.mean(rel_diff_slices, axis=0)
    # PDF of Relative Error (log-binned)
    error_bins = np.logspace(-6, 1, 101)
    rel_diff_pdf, rel_diff_bins = np.histogram(rel_diff_slices.flatten(), bins=error_bins, density=True)
    rel_diff_bin_centers = 0.5 * (rel_diff_bins[1:] + rel_diff_bins[:-1])
    
    # Derivatives
    dslices = deriv(slices)
    dc_mean = np.mean(dslices, axis=0)
    dc_sq_mean = np.mean(dslices**2, axis=0)
    dc_std = np.std(dslices, axis=0)
    ddiff_slices = np.abs(dorig_slices - dslices)
    rel_ddiff_slices = np.where(np.abs(dorig_slices) > 0, ddiff_slices / np.abs(dorig_slices), ddiff_slices)
    rel_ddiff_mean_prof = np.mean(rel_ddiff_slices, axis=0)
    rel_ddiff_pdf, rel_ddiff_bins = np.histogram(rel_ddiff_slices.flatten(), bins=error_bins, density=True)
    rel_ddiff_bin_centers = 0.5 * (rel_ddiff_bins[1:] + rel_ddiff_bins[:-1])
    
    # Increments
    delta_slices_torch = compute_increments(comp_data[m]['slices_torch'], axis=1, r=increment_r, stride=downsample, inverse_stride=inverse_stride)
    delta_slices = delta_slices_torch.cpu().numpy()
    d_sq_mean = np.mean(delta_slices**2, axis=0)
    d_qu_mean = np.mean(delta_slices**4, axis=0)
    s2_avg = np.mean(d_sq_mean)
    s4_avg = np.mean(d_qu_mean)
    s2sq_avg = s2_avg**2
    
    # Flatness
    flatness_signal = d_qu_mean / s2sq_avg
    flatness_avg = s4_avg / s2sq_avg
    
    # Conditional Flatness
    if filter_per_point:
        flatness_2d = delta_slices**4 / s2sq_avg
        mask = flatness_2d <= flatness_threshold
        n_rem = mask.sum()
        n_tot = mask.size
        flatness_filtered_avg = np.mean(flatness_2d[mask])
        y_counts = np.sum(~mask, axis=0)
        affected_x = np.where(y_counts > 0)[0]
        removed_x = [f"{x_idx * downsample} ({y_counts[x_idx]})" for x_idx in affected_x]
    else:
        mask = flatness_signal <= flatness_threshold
        n_rem = mask.sum()
        n_tot = mask.size
        flatness_filtered_avg = np.mean(flatness_signal[mask])
        removed_x = [f"{x_val * downsample} ({len(slice_y_list)})" for x_val in np.where(~mask)[0]]

    perc_rem = 100.0 * (n_tot - n_rem) / n_tot

    # Store stats
    comp_data[m].update({
        'mean': c_mean, 'std': c_std, 'global_mean': c_global_mean,
        'diff_mean': np.mean(diff_slices, axis=0),
        'rel_diff_prof': rel_diff_mean_prof,
        'rel_diff_pdf': rel_diff_pdf, 'rel_diff_bins': rel_diff_bin_centers,
        'dc_mean': dc_mean, 'dc_sq_mean': dc_sq_mean, 'dc_std': dc_std,
        'ddiff_mean': np.mean(ddiff_slices, axis=0),
        'rel_ddiff_prof': rel_ddiff_mean_prof,
        'rel_ddiff_pdf': rel_ddiff_pdf, 'rel_ddiff_bins': rel_ddiff_bin_centers,
        'd_mean': np.mean(delta_slices, axis=0),
        'd_sq_mean': d_sq_mean, 'd_qu_mean': d_qu_mean,
        's2_avg': s2_avg, 's4_avg': s4_avg, 's2sq_avg': s2sq_avg,
        'flatness_signal': flatness_signal, 'flatness_avg': flatness_avg,
        'flatness_filtered_avg': flatness_filtered_avg,
        'perc_removed': perc_rem,
        'removed_x': removed_x
    })

# Define colors for plotting
method_colors = {slice_method_1: 'red', slice_method_2: 'blue'}
method_styles = {slice_method_1: '--', slice_method_2: '-.'}
x = np.arange(f_slice_orig.shape[1])

if plot_fig1:
    x_d = x
    def find_large_error_positions(diff, x_vals, threshold=0.4):
        """Return x indices where the absolute difference exceeds `threshold`."""
        mask = diff > threshold
        return x_vals[mask]

    for m in methods:
        ext_x = find_large_error_positions(comp_data[m]['diff_mean'], x, threshold=0.4)
        print(f"[{comp_data[m]['label']}] x positions where mean |diff| > 0.4: {ext_x}")
        
        ext_dx = find_large_error_positions(comp_data[m]['ddiff_mean'], x_d, threshold=0.4)
        print(f"[{comp_data[m]['label']}] x positions where mean |d(diff)/dx| > 0.4: {ext_dx}")

    # --- Plot Figure 1 (Slices and Derivatives) ---
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))

    # Row 1: field values
    axes[0, 0].plot(x, orig_mean, 'k-', linewidth=1.5, label=f'Original (avg={orig_global_mean:.3f})', alpha=0.9)
    for m in methods:
        axes[0, 0].plot(x, comp_data[m]['mean'], color=method_colors[m], linestyle=method_styles[m], 
                        linewidth=1.0, label=f"{comp_data[m]['label']} (avg={comp_data[m]['global_mean']:.3f})", alpha=0.8)
    axes[0, 0].set_title(f'Overlapped $\\theta$ (Averaged y={slice_y_start}:{slice_y_end})', fontsize=14)
    axes[0, 0].set_xlabel('x', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Zoomed Overlapped signals
    axes[0, 1].plot(x, orig_mean, 'k-', linewidth=1.5, label='Original', alpha=0.9)
    for m in methods:
        axes[0, 1].plot(x, comp_data[m]['mean'], color=method_colors[m], linestyle=method_styles[m], 
                        linewidth=1.0, label=comp_data[m]['label'], alpha=0.8)
    axes[0, 1].set_title(f'Zoomed Overlapped $\\theta$ (x=[{zoom_x_min},{zoom_x_max}])', fontsize=14)
    axes[0, 1].set_xlabel('x', fontsize=12)
    axes[0, 1].set_xlim(zoom_x_min, zoom_x_max)
    axes[0, 1].grid(True, alpha=0.3)

    # PDF of Relative Error (Field) vs Grid Plot
    if plot_pdf_rel_error:
        for m in methods:
            axes[0, 2].step(comp_data[m]['rel_diff_bins'], comp_data[m]['rel_diff_pdf'], color=method_colors[m], 
                            where='mid', label=comp_data[m]['label'])
        axes[0, 2].set_xscale('log')
        axes[0, 2].set_yscale('log')
        axes[0, 2].set_title('PDF of Relative Error $|O-C|/|O|$', fontsize=14)
        axes[0, 2].set_xlabel('Relative Error', fontsize=12)
        axes[0, 2].set_ylabel('Probability Density', fontsize=12)
    else:
        for m in methods:
            axes[0, 2].plot(x, comp_data[m]['rel_diff_prof'], color=method_colors[m], 
                            linestyle=method_styles[m], label=comp_data[m]['label'])
        axes[0, 2].set_title('Relative Error Mag. $|O-C|/|O|$', fontsize=14)
        axes[0, 2].set_xlabel('x', fontsize=12)
        # axes[0, 2].set_ylim(0, 1.0) # removed for automatic scaling
    axes[0, 2].grid(True, alpha=0.3, which='both')
    axes[0, 2].legend()

    # Row 2: x-derivatives
    axes[1, 0].plot(x_d, dorig_mean, 'k-', linewidth=1.5, label='Original (mean)', alpha=0.9)
    for m in methods:
        axes[1, 0].plot(x_d, comp_data[m]['dc_mean'], color=method_colors[m], linestyle=method_styles[m], 
                        linewidth=1.0, label=comp_data[m]['label'], alpha=0.8)
    axes[1, 0].set_title(f'Overlapped $\\partial_x\\theta$ (Averaged y={slice_y_start}:{slice_y_end})', fontsize=14)
    axes[1, 0].set_xlabel('x', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Zoomed Derivative
    axes[1, 1].plot(x_d, dorig_mean, 'k-', linewidth=1.5, label='Original', alpha=0.9)
    for m in methods:
        axes[1, 1].plot(x_d, comp_data[m]['dc_mean'], color=method_colors[m], linestyle=method_styles[m], 
                        linewidth=1.0, label=comp_data[m]['label'], alpha=0.8)
    axes[1, 1].set_title(f'Zoomed Overlapped $\\partial_x\\theta$ (x=[{zoom_x_min},{zoom_x_max}])', fontsize=14)
    axes[1, 1].set_xlabel('x', fontsize=12)
    axes[1, 1].set_xlim(zoom_x_min, zoom_x_max)
    axes[1, 1].grid(True, alpha=0.3)

    # PDF of Relative Derivative Error vs Grid Plot
    if plot_pdf_rel_error:
        for m in methods:
            axes[1, 2].step(comp_data[m]['rel_ddiff_bins'], comp_data[m]['rel_ddiff_pdf'], color=method_colors[m], 
                            where='mid', label=comp_data[m]['label'])
        axes[1, 2].set_xscale('log')
        axes[1, 2].set_yscale('log')
        axes[1, 2].set_title('PDF of Rel. Deriv. Error', fontsize=14)
        axes[1, 2].set_ylabel('Probability Density', fontsize=12)
    else:
        for m in methods:
            axes[1, 2].plot(x_d, comp_data[m]['rel_ddiff_prof'], color=method_colors[m], 
                            linestyle=method_styles[m], label=comp_data[m]['label'])
        axes[1, 2].set_title('Rel. Deriv. Error $|dO-dC|/|dO|$', fontsize=14)
        # axes[1, 2].set_ylim(0, 5.0) # removed for automatic scaling
    axes[1, 2].set_xlabel('Relative Error', fontsize=12)
    axes[1, 2].grid(True, alpha=0.3, which='both')
    axes[1, 2].legend()

    plt.suptitle(f'1D Slice Average Analysis — config={slice_config:05d}, BD={slice_BD_MAX}, y-range={slice_y_start}-{slice_y_end}', fontsize=16, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


# --- Reporting ---
filter_type = "Per-Point (2D)" if filter_per_point else "Ensemble-Averaged (Spatial)"
print(f"\n--- Comparative Moment Analysis (y-range={slice_y_start}:{slice_y_end}) ---")
print(f"Filtering Strategy: {filter_type}")
headers = f"{'Method':<25} | {'<dx^2>':<10} | {'<dx^4>':<10} | {'Flatness F':<10} | {'F (filtered)':<12} | {'% Removed':<10}"
print(headers)
print("-" * len(headers))
print(f"{'Original':<25} | {S2_orig_avg:<10.2e} | {S4_orig_avg:<10.2e} | {flatness_orig_avg:<10.2f} | {flatness_orig_filtered_avg:<12.2f} | {perc_removed_orig:<10.2f}%")
for m in methods:
    d = comp_data[m]
    print(f"{d['label']:<25} | {d['s2_avg']:<10.2e} | {d['s4_avg']:<10.2e} | {d['flatness_avg']:<10.2f} | {d['flatness_filtered_avg']:<12.2f} | {d['perc_removed']:<10.2f}%")
print(f"\nMax original F point: {max_val_orig:.2f} at (x={max_x_coord}, y={max_y_coord})")
print(f"--- Extreme Flatness (Threshold > {flatness_threshold:.1f}), 2x maximal one ---")
if filter_per_point == False:
    print ("(Remove points only if the y-average is above the threshold)")
print(f"Original removed x-positions: {removed_x_orig}")
for m in methods:
    rem_x = comp_data[m]['removed_x']
    print(f"{comp_data[m]['label']} removed x-positions: {rem_x}")
    print(f"Total unique x-positions affected: {len(rem_x)}")
    
# Report locations where <(delta theta)^2> exceeds incr_tresh
print("\n --- Extreme increments ---")
print(f"multiples of 64:{x_inc[63::64].tolist()}")
for m in methods:
    large_incr_indices = np.where(comp_data[m]['d_sq_mean'] > incr_tresh)[0]
    large_incr_x = x_inc[large_incr_indices]
    if len(large_incr_x) > 0:
        print(f"{comp_data[m]['label']} positions with <(delta theta)^2> > {incr_tresh:.1e}: {large_incr_x.tolist()}")
    else:
        print(f"{comp_data[m]['label']} has no positions with <(delta theta)^2> > {incr_tresh:.1e}")

# Report locations where <(delta theta)^2> is below inf_incr_thresh
print("\n --- Extremely small increments ---")
for m in methods:
    small_incr_indices = np.where(comp_data[m]['d_sq_mean'] < inf_incr_thresh)[0]
    small_incr_x = x_inc[small_incr_indices]
    if len(small_incr_x) > 0:
        print(f"{comp_data[m]['label']} positions with <(delta theta)^2> < {inf_incr_thresh:.1e}: {small_incr_x.tolist()}")
    else:
        print(f"{comp_data[m]['label']} has no positions with <(delta theta)^2> < {inf_incr_thresh:.1e}")

# --- Plot Figure 2 (Increment Moments) ---
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 6))

# Plot 1: dx^2
ax1.plot(x_inc, delta_orig_sq_mean, 'k-', linewidth=1.2, label=f'Original (avg={S2_orig_avg:.2e})', alpha=0.8)
for m in methods:
    ax1.plot(x_inc, comp_data[m]['d_sq_mean'], color=method_colors[m], linestyle=method_styles[m],
             linewidth=1.0, label=f"{comp_data[m]['label']} (avg={comp_data[m]['s2_avg']:.2e})", alpha=0.8)
ax1.set_title(f'$(\\delta\\theta)^2$ signal ($r={increment_r}$, stride={downsample}, skip={inverse_stride})', fontsize=14)
ax1.set_xlabel('x', fontsize=12)
if fig2_log_y:
    ax1.set_yscale('log')
ax1.grid(True, alpha=0.3, which='both')
ax1.legend(loc='upper right', framealpha=0.9)

# Plot 2: dx^4
ax2.plot(x_inc, delta_orig_qu_mean, 'k-', linewidth=1.2, label=f'Original (avg={S4_orig_avg:.2e})', alpha=0.8)
for m in methods:
    ax2.plot(x_inc, comp_data[m]['d_qu_mean'], color=method_colors[m], linestyle=method_styles[m],
             linewidth=1.0, label=f"{comp_data[m]['label']} (avg={comp_data[m]['s4_avg']:.2e})", alpha=0.8)
ax2.set_title(f'$(\\delta\\theta)^4$ signal ($r={increment_r}$, stride={downsample}, skip={inverse_stride})', fontsize=14)
ax2.set_xlabel('x', fontsize=12)
if fig2_log_y:
    ax2.set_yscale('log')
ax2.grid(True, alpha=0.3, which='both')
ax2.legend(loc='upper right', framealpha=0.9)

# Plot 3: flatness
ax3.plot(x_inc, delta_orig_qu_mean/S2sq_orig_avg, 'k-', linewidth=1.2, label=f'Original (F={S4_orig_avg/S2sq_orig_avg:.2f})', alpha=0.8)
for m in methods:
    ax3.plot(x_inc, comp_data[m]['flatness_signal'], color=method_colors[m], linestyle=method_styles[m],
             linewidth=1.0, label=f"{comp_data[m]['label']} (F={comp_data[m]['flatness_avg']:.2f})", alpha=0.8)
ax3.set_title(f'Flatness signal $(\\delta\\theta)^4 / \\langle\\delta\\theta^2\\rangle^2$ ($r={increment_r}$, stride={downsample}, skip={inverse_stride})', fontsize=14)
ax3.set_xlabel('x', fontsize=12)
if fig2_log_y:
    ax3.set_yscale('log')
else:
    pass # ax3.set_ylim(0, flatness_threshold) # commented out by user previously
ax3.grid(True, alpha=0.3, which='both')
ax3.legend(loc='upper right', framealpha=0.9)

plt.suptitle(f'Moment Analysis - Unfiltered plots - ($r={increment_r}$, stride={downsample}, skip={inverse_stride}) — config={slice_config:05d}, BD={slice_BD_MAX}, averaged over y', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

# --- Figure 3 Analysis (Flatness vs r) ---
print(f"\n--- Scale-Dependent Flatness Analysis (r={r_values_fig3}) ---")
F_vs_r = {'Original': []}
for m in methods: F_vs_r[m] = []
F_vs_r_unfilt = {'Original': []}
for m in methods: F_vs_r_unfilt[m] = []

# Dictionary to store PDF data for various r increments
delta_pdfs = {}

# Initialize common bins for normalized PDF (consistent with Section 2)
bins_pdf = np.arange(-20, 20, 0.1)
bin_centers_pdf = 0.5 * (bins_pdf[:-1] + bins_pdf[1:])

for r in r_values_fig3:
    # Original: use compute_increments for better consistency (no periodic wrap-around)
    delta_orig_torch = compute_increments(f_slice_orig[slice_y_list, :], axis=1, r=r, stride=downsample, inverse_stride=inverse_stride)
    delta_orig = delta_orig_torch.cpu().numpy()
    
    if r in [1, 2, 4, 8, 16]:
        # Compute Unfiltered and Filtered PDFs for current scale r
        std_orig_r = np.std(delta_orig)
        delta_orig_norm_r = delta_orig / std_orig_r
        
        # Unfiltered PDF
        cnt_orig_unfilt, _ = np.histogram(delta_orig_norm_r, bins=bins_pdf, density=True)
        delta_pdfs[f'Original_Unfilt_r{r}'] = {'bins': bin_centers_pdf, 'pdf': cnt_orig_unfilt}

        # Filtered PDF for this scale
        s2sq_orig_avg_r_filt = (np.mean(delta_orig**2))**2
        f2d_orig_r_filt = delta_orig**4 / s2sq_orig_avg_r_filt
        # Using scale-dependent threshold for Row 2
        thresh_r_pdf = flatness_threshold_multiplier * np.max(f2d_orig_r_filt)
        
        # Original signal in the "Filtered" plot is actually kept Unfiltered as per request
        cnt_orig_filt, _ = np.histogram(delta_orig_norm_r, bins=bins_pdf, density=True)
        delta_pdfs[f'Original_Filt_r{r}'] = {'bins': bin_centers_pdf, 'pdf': cnt_orig_filt}
        
        # We still need thresh_1 for the default r=1 methods filtering if needed
        if r == 1:
            thresh_1 = thresh_r_pdf
    s2_orig_avg_r = np.mean(delta_orig**2)
    s4_orig_avg_r = np.mean(delta_orig**4)
    s2sq_orig_avg_r = s2_orig_avg_r**2
    
    f2d_orig_r = delta_orig**4 / s2sq_orig_avg_r
    # Threshold is multiplier * the peak point-wise flatness AT THIS SCALE
    thresh_r = flatness_threshold_multiplier * np.max(f2d_orig_r)
    
    F_orig_unfilt_r = np.mean(f2d_orig_r)
    F_vs_r_unfilt['Original'].append(F_orig_unfilt_r)
    
    # Original is NOT filtered
    F_orig_r = np.mean(f2d_orig_r if filter_per_point else np.mean(delta_orig**4, axis=0) / s2sq_orig_avg_r)
    F_vs_r['Original'].append(F_orig_r)
    
    for m in methods:
        field_m, _ = load_compressed_field(m, slice_config, slice_BD_MAX, base_dir, f_slice_orig)
        delta_m_torch = compute_increments(field_m[slice_y_list, :], axis=1, r=r, stride=downsample, inverse_stride=inverse_stride)
        delta_m = delta_m_torch.cpu().numpy()
        
        s2_m_avg_r = np.mean(delta_m**2)
        s2sq_m_avg_r = s2_m_avg_r**2
        
        if r in [1, 2, 4, 8, 16]:
            # Normalize increments by their own standard deviation
            std_m_r = np.std(delta_m)
            delta_m_norm_r = delta_m / std_m_r
            
            # Unfiltered PDF
            cnt_m_unfilt, _ = np.histogram(delta_m_norm_r, bins=bins_pdf, density=True)
            delta_pdfs[f"{m}_Unfilt_r{r}"] = {'bins': bin_centers_pdf, 'pdf': cnt_m_unfilt}
            
            # Filtered PDF for this scale
            # We use scale-dependent thresh_r (or thresh_1 if r=1)
            cur_thresh = thresh_1 if r == 1 else thresh_r
            
            if filter_per_point:
                f2d_m_r_filt = delta_m**4 / s2sq_m_avg_r
                mask_m_r_filt = f2d_m_r_filt <= cur_thresh
                cnt_m_filt, _ = np.histogram(delta_m_norm_r[mask_m_r_filt], bins=bins_pdf, density=True)
            else:
                f_prof_m_r_filt = np.mean(delta_m**4, axis=0) / s2sq_m_avg_r
                mask_m_r_filt = f_prof_m_r_filt <= cur_thresh
                mask_m_r_2d = np.tile(mask_m_r_filt, (delta_m.shape[0], 1))
                cnt_m_filt, _ = np.histogram(delta_m_norm_r[mask_m_r_2d], bins=bins_pdf, density=True)
            
            delta_pdfs[f"{m}_Filt_r{r}"] = {'bins': bin_centers_pdf, 'pdf': cnt_m_filt}
        
        f2d_m_r = delta_m**4 / s2sq_m_avg_r
        F_m_unfilt_r = np.mean(f2d_m_r)
        F_vs_r_unfilt[m].append(F_m_unfilt_r)

        if filter_per_point:
            mask_m_r = f2d_m_r <= thresh_r
            F_m_r = np.mean(f2d_m_r[mask_m_r])
        else:
            f_prof_m_r = np.mean(delta_m**4, axis=0) / s2sq_m_avg_r
            mask_m_r = f_prof_m_r <= thresh_r
            F_m_r = np.mean(f_prof_m_r[mask_m_r])
        F_vs_r[m].append(F_m_r)

# --- Plot Figure 3 (Intermittency Summary: Flatness and Multi-scale PDFs) ---
fig3, axes = plt.subplots(2, 4, figsize=(32, 16))
(fax1, fax2, fax3, fax4) = axes[0]
(fax5, fax6, fax7, fax8) = axes[1]

# Row 1: Flatness Scaling and r=1 PDFs
# Subplot 3.1: Unfiltered Flatness Scaling
fax1.plot(r_values_fig3, F_vs_r_unfilt['Original'], 'ko-', linewidth=2, markersize=8, label='Original (Unfilt)')
for m in methods:
    fax1.plot(r_values_fig3, F_vs_r_unfilt[m], color=method_colors[m], marker='s', 
             linestyle=method_styles[m], linewidth=1.5, markersize=6, label=comp_data[m]['label'])

fax1.set_xscale('log')
fax1.set_xlabel('Separation distance $r$', fontsize=12)
fax1.set_ylabel('Flatness Factor $F(r)$', fontsize=12)
fax1.set_title(f'Unfiltered Flatness Scaling $F(r)$', fontsize=14)
fax1.grid(True, which='both', alpha=0.3)
fax1.legend()

# Subplot 3.2: Filtered Flatness Scaling
fax2.plot(r_values_fig3, F_vs_r['Original'], 'ko-', linewidth=2, markersize=8, label='Original (Unfilt)')
for m in methods:
    fax2.plot(r_values_fig3, F_vs_r[m], color=method_colors[m], marker='s', 
             linestyle=method_styles[m], linewidth=1.5, markersize=6, label=comp_data[m]['label'])

fax2.set_xscale('log')
fax2.set_xlabel('Separation distance $r$', fontsize=12)
fax2.set_ylabel('Flatness Factor $F(r)$', fontsize=12)
fax2.set_title(f'Filtered Flatness Scaling $F(r)$', fontsize=14)
fax2.grid(True, which='both', alpha=0.3)
fax2.legend()

# Subplot 3.3: Unfiltered Increment PDF at r=1
fax3.step(delta_pdfs['Original_Unfilt_r1']['bins'], delta_pdfs['Original_Unfilt_r1']['pdf'], 
          'k-', where='mid', linewidth=2.5, label='Original (Unfiltered)')

for m in methods:
    fax3.step(delta_pdfs[f"{m}_Unfilt_r1"]['bins'], delta_pdfs[f"{m}_Unfilt_r1"]['pdf'], 
              color=method_colors[m], linestyle=method_styles[m], where='mid', 
              linewidth=1.5, label=f"{comp_data[m]['label']} (Unfilt)")

fax3.set_yscale('log')
fax3.set_xlabel('Normalized increment $\\delta\\theta / \\sigma_{\\delta\\theta}$', fontsize=12)
fax3.set_ylabel('Probability Density', fontsize=12)
fax3.set_title(f'Unfiltered Increment PDF at $r=1$', fontsize=14)
fax3.grid(True, which='both', alpha=0.3)
fax3.legend()

# Subplot 3.4: Filtered Increment PDF at r=1
fax4.step(delta_pdfs['Original_Filt_r1']['bins'], delta_pdfs['Original_Filt_r1']['pdf'], 
          'k-', where='mid', linewidth=2.5, label='Original (Unfiltered)')

for m in methods:
    fax4.step(delta_pdfs[f"{m}_Filt_r1"]['bins'], delta_pdfs[f"{m}_Filt_r1"]['pdf'], 
              color=method_colors[m], linestyle=method_styles[m], where='mid', 
              linewidth=1.5, label=f"{comp_data[m]['label']} (Filt)")

fax4.set_yscale('log')
fax4.set_xlabel('Normalized increment $\\delta\\theta / \\sigma_{\\delta\\theta}$', fontsize=12)
fax4.set_ylabel('Probability Density', fontsize=12)
fax4.set_title(f'Filtered Increment PDF at $r=1$ (Orig. Unfilt.)', fontsize=14)
fax4.grid(True, which='both', alpha=0.3)
fax4.legend()

# Row 2: Unfiltered PDFs for r=2, 4, 8, 16
for i, r_val in enumerate([2, 4, 8, 16]):
    ax_cur = [fax5, fax6, fax7, fax8][i]
    ax_cur.step(delta_pdfs[f'Original_Unfilt_r{r_val}']['bins'], delta_pdfs[f'Original_Unfilt_r{r_val}']['pdf'], 
                'k-', where='mid', linewidth=2.5, label='Original (Unfilt)')
    for m in methods:
        ax_cur.step(delta_pdfs[f"{m}_Unfilt_r{r_val}"]['bins'], delta_pdfs[f"{m}_Unfilt_r{r_val}"]['pdf'], 
                    color=method_colors[m], linestyle=method_styles[m], where='mid', 
                    linewidth=1.5, label=f"{comp_data[m]['label']} (Unfilt)")
    
    ax_cur.set_yscale('log')
    ax_cur.set_xlabel('Normalized increment $\\delta\\theta / \\sigma_{\\delta\\theta}$', fontsize=12)
    ax_cur.set_ylabel('Probability Density', fontsize=12)
    ax_cur.set_title(f'Unfiltered Increment PDF at $r={r_val}$', fontsize=14)
    ax_cur.grid(True, which='both', alpha=0.3)
    ax_cur.legend()

plt.suptitle(f'Intermittency & Statistics — config={slice_config:05d}, stride={downsample}, skip={inverse_stride}, strategy={"Per-Point" if filter_per_point else "Ensemble"}', fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()




# %% Zoomed comparison: theta, derivative, increment, squared derivative, and increment squared
# Col 1: zoomed overlap of θ
# Col 2: zoomed overlap of ∂_x θ  (= Fig 1, plot 5)
# Col 3: δθ signal                 (increment itself, y-averaged)
# Col 4: (∂_x θ)²                 (= square of Col 2)
# Col 5: (δθ)² signal              (= Fig 2, plot 1, zoomed to same x-window)

# --- Coordinate mask for calculating unified y-limits ---
zoom_mask = (x >= zoom_x_min) & (x <= zoom_x_max)
zoom_mask_inc = (x_inc >= zoom_x_min) & (x_inc <= zoom_x_max)

# Initialize data collection lists
all_y_vals_theta = []
all_y_vals_sig = []
all_y_vals_sq = []

# Precompute mean of square for original derivative for the plot
dorig_sq_mean = np.mean(dorig_slices**2, axis=0)

# Collect original data
all_y_vals_theta.extend(orig_mean[zoom_mask].tolist())
all_y_vals_sig.extend(dorig_mean[zoom_mask].tolist())
all_y_vals_sig.extend(delta_orig_mean[zoom_mask_inc].tolist())
all_y_vals_sq.extend(dorig_sq_mean[zoom_mask].tolist())
all_y_vals_sq.extend(delta_orig_sq_mean[zoom_mask_inc].tolist())

# Collect compressed methods data
for m in methods:
    all_y_vals_theta.extend(comp_data[m]['mean'][zoom_mask].tolist())
    all_y_vals_sig.extend(comp_data[m]['dc_mean'][zoom_mask].tolist())
    all_y_vals_sig.extend(comp_data[m]['d_mean'][zoom_mask_inc].tolist())
    all_y_vals_sq.extend(comp_data[m]['dc_sq_mean'][zoom_mask].tolist())
    all_y_vals_sq.extend(comp_data[m]['d_sq_mean'][zoom_mask_inc].tolist())

y_min_theta, y_max_theta = np.min(all_y_vals_theta), np.max(all_y_vals_theta)
yr_theta = y_max_theta - y_min_theta
y_lims_theta = (y_min_theta - 0.05 * yr_theta, y_max_theta + 0.05 * yr_theta)

y_min_sig, y_max_sig = np.min(all_y_vals_sig), np.max(all_y_vals_sig)
yr_sig = y_max_sig - y_min_sig
y_lims_sig = (y_min_sig - 0.05 * yr_sig, y_max_sig + 0.05 * yr_sig)

y_min_sq, y_max_sq = np.min(all_y_vals_sq), np.max(all_y_vals_sq)
yr_sq = y_max_sq - y_min_sq
y_lims_sq = (y_min_sq - 0.05 * yr_sq, y_max_sq + 0.05 * yr_sq)

fig_zoom, axes_zoom = plt.subplots(1, 5, figsize=(40, 6))

# --- Col 1: zoomed θ ---
axes_zoom[0].plot(x, orig_mean, 'k-o', linewidth=1.5, label='Original', alpha=0.9, markersize=4)
for m in methods:
    axes_zoom[0].plot(x, comp_data[m]['mean'],
                      color=method_colors[m], linestyle=method_styles[m], marker='o',
                      linewidth=1.0, label=comp_data[m]['label'], alpha=0.8, markersize=3)
axes_zoom[0].set_xlim(zoom_x_min, zoom_x_max)
axes_zoom[0].set_ylim(y_lims_theta)
axes_zoom[0].set_title(f'Zoomed $\\theta$ (x=[{zoom_x_min},{zoom_x_max}])', fontsize=14)
axes_zoom[0].set_xlabel('x', fontsize=12)
axes_zoom[0].grid(True, alpha=0.3)
axes_zoom[0].legend()

# --- Col 2: zoomed ∂_x θ ---
axes_zoom[1].plot(x, dorig_mean, 'k-o', linewidth=1.5, label='Original', alpha=0.9, markersize=4)
for m in methods:
    axes_zoom[1].plot(x, comp_data[m]['dc_mean'],
                      color=method_colors[m], linestyle=method_styles[m], marker='o',
                      linewidth=1.0, label=comp_data[m]['label'], alpha=0.8, markersize=3)
axes_zoom[1].set_xlim(zoom_x_min, zoom_x_max)
axes_zoom[1].set_ylim(y_lims_sig)
axes_zoom[1].set_title(f'Zoomed $\\partial_x\\theta$ (x=[{zoom_x_min},{zoom_x_max}])', fontsize=14)
axes_zoom[1].set_xlabel('x', fontsize=12)
axes_zoom[1].grid(True, alpha=0.3)
axes_zoom[1].legend()

# --- Col 3: δθ (increment itself, y-averaged) ---
axes_zoom[2].plot(x_inc[zoom_mask_inc], delta_orig_mean[zoom_mask_inc],
                  'k-o', linewidth=1.5, label='Original', alpha=0.9, markersize=4)
for m in methods:
    axes_zoom[2].plot(x_inc[zoom_mask_inc], comp_data[m]['d_mean'][zoom_mask_inc],
                      color=method_colors[m], linestyle=method_styles[m], marker='o',
                      linewidth=1.0, label=comp_data[m]['label'], alpha=0.8, markersize=3)
axes_zoom[2].axhline(0, color='gray', linewidth=0.8, linestyle=':')
axes_zoom[2].set_ylim(y_lims_sig)
axes_zoom[2].set_title(f'Zoomed $\\delta\\theta$ ($r={increment_r}$, x=[{zoom_x_min},{zoom_x_max}])', fontsize=14)
axes_zoom[2].set_xlabel('x', fontsize=12)
axes_zoom[2].grid(True, alpha=0.3, which='both')
axes_zoom[2].legend()

# --- Col 4: (∂_x θ)² ---
axes_zoom[3].plot(x, dorig_sq_mean, 'k-o', linewidth=1.5, label='Original', alpha=0.9, markersize=4)
for m in methods:
    axes_zoom[3].plot(x, comp_data[m]['dc_sq_mean'],
                      color=method_colors[m], linestyle=method_styles[m], marker='o',
                      linewidth=1.0, label=comp_data[m]['label'], alpha=0.8, markersize=3)
axes_zoom[3].set_xlim(zoom_x_min, zoom_x_max)
axes_zoom[3].set_ylim(y_lims_sq)
axes_zoom[3].set_title(f'Zoomed $(\\partial_x\\theta)^2$ (x=[{zoom_x_min},{zoom_x_max}])', fontsize=14)
axes_zoom[3].set_xlabel('x', fontsize=12)
axes_zoom[3].grid(True, alpha=0.3)
axes_zoom[3].legend()

# --- Col 5: (δθ)² zoomed to the same x-window ---
axes_zoom[4].plot(x_inc[zoom_mask_inc], delta_orig_sq_mean[zoom_mask_inc],
                  'k-o', linewidth=1.5, label='Original', alpha=0.9, markersize=4)
for m in methods:
    axes_zoom[4].plot(x_inc[zoom_mask_inc], comp_data[m]['d_sq_mean'][zoom_mask_inc],
                      color=method_colors[m], linestyle=method_styles[m], marker='o',
                      linewidth=1.0, label=comp_data[m]['label'], alpha=0.8, markersize=3)
axes_zoom[4].set_ylim(y_lims_sq)
axes_zoom[4].set_title(f'Zoomed $(\\delta\\theta)^2$ ($r={increment_r}$, x=[{zoom_x_min},{zoom_x_max}])', fontsize=14)
axes_zoom[4].set_xlabel('x', fontsize=12)
axes_zoom[4].grid(True, alpha=0.3, which='both')
axes_zoom[4].legend()

plt.suptitle(f'Zoomed Derivative vs Increments — config={slice_config:05d}, BD={slice_BD_MAX}, stride={downsample}, skip={inverse_stride}, y=[{slice_y_start},{slice_y_end}]', fontsize=15, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()



# %% Flatness for different downsampling factors (stride) to check if the stats is independent of the stride

flat_config   = slice_config   # snapshot to use
flat_BD       = slice_BD_MAX   # bond dimension
flat_method_1 = slice_method_1 # first compressed method  (e.g. 'hybrid_N12_p4')
flat_method_2 = slice_method_2 # second compressed method (e.g. 'hybrid_N12_p4_int64')
stride_list   = [1, 2, 3, 4, 5, 8, 16]
r_list_flat   = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]

# --- Load fields (original already in f_slice_orig) ---
f_flat_m1, label_flat_m1 = load_compressed_field(flat_method_1, flat_config, flat_BD, base_dir, f_slice_orig)
f_flat_m2, label_flat_m2 = load_compressed_field(flat_method_2, flat_config, flat_BD, base_dir, f_slice_orig)

# --- Helper: flatness curve for one field and one stride ---
def compute_flatness_vs_r(field, r_list, stride):
    sf = compute_structure_functions([field], r_list, p_list=[2, 4], axis=1, stride=stride, normalize=False)
    return sf[4] / sf[2]**2

# --- Plot ---
stride_colors  = plt.cm.plasma(np.linspace(0.1, 0.9, len(stride_list)))
stride_markers = ['o', 's', '^', 'D', 'v', 'P', 'X']  # one per stride entry

# Pre-compute original flatness at stride=1 (reference line for compressed panels)
F_orig_ref = compute_flatness_vs_r(f_slice_orig, r_list_flat, stride=1).numpy()

fig_flat_stride, axes_flat_stride = plt.subplots(1, 3, figsize=(21, 6), sharey=True)

signals_flat = [
    ('Original',    f_slice_orig),
    (label_flat_m1, f_flat_m1),
    (label_flat_m2, f_flat_m2),
]

for ax, (sig_label, field) in zip(axes_flat_stride, signals_flat):
    for stride, color, marker in zip(stride_list, stride_colors, stride_markers):
        F = compute_flatness_vs_r(field, r_list_flat, stride)
        ax.loglog(r_list_flat, F.numpy(), linestyle='-', marker=marker, color=color,
                  linewidth=1.5, markersize=8, label=f'stride={stride}')
    ax.loglog(r_list_flat, F_orig_ref, linestyle='--', color='black',
              linewidth=2.0, marker='o', markersize=8, label='Original (stride=1)')
    ax.set_title(sig_label, fontsize=13, fontweight='bold')
    ax.set_xlabel('Separation $r$', fontsize=12)
    ax.grid(True, which='both', alpha=0.3)
    ax.legend(fontsize=10)

axes_flat_stride[0].set_ylabel('Flatness $F(r) = S_4 / S_2^2$', fontsize=12)

plt.suptitle(f'Flatness vs stride \u2014 config={flat_config:05d}, BD={flat_BD}', fontsize=14, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()

















































# config = 59
# N = 12
# BD_MAX = 100

# field = "vx"
# vx_hybrid = torch.load(f'{cfg.HYB_DIR}/{field}000{config}_hybrid_mps_N{N}_bd_equiv{BD_MAX}.tens')
# vx_standard = torch.load(f'{cfg.STD_DIR}/{field}000{config}_standard_mps_bd{BD_MAX}.tens')
# vx_hybrid_mps_compression = torch.load(f'{cfg.HYB_DIR}/{field}000{config}_hybrid_mps_N{N}_bd_equiv{BD_MAX}.compression')
# vx_standard_mps_compression = torch.load(f'{cfg.STD_DIR}/{field}000{config}_standard_mps_bd{BD_MAX}.compression')

# field = "vy"
# vy_hybrid = torch.load(f'{cfg.HYB_DIR}/{field}000{config}_hybrid_mps_N{N}_bd_equiv{BD_MAX}.tens')
# vy_standard = torch.load(f'{cfg.STD_DIR}/{field}000{config}_standard_mps_bd{BD_MAX}.tens')
# vy_hybrid_mps_compression = torch.load(f'{cfg.HYB_DIR}/{field}000{config}_hybrid_mps_N{N}_bd_equiv{BD_MAX}.compression')
# vy_standard_mps_compression = torch.load(f'{cfg.STD_DIR}/{field}000{config}_standard_mps_bd{BD_MAX}.compression')



# # Wavenumbers
# N = 4096
# kx = torch.cat((torch.arange(0, N//2), torch.arange(-N//2, 0)))
# KY, KX = torch.meshgrid(kx, kx, indexing='ij')

# # Velocities in Fourier space
# vxf_hybrid = torch.fft.fft2(vx_hybrid)
# vyf_hybrid = torch.fft.fft2(vy_hybrid)

# vxf_standard = torch.fft.fft2(vx_standard)
# vyf_standard = torch.fft.fft2(vy_standard)

# # Velocity gradients in Fourier space
# dvxdxf_hybrid = 1j * KX * vxf_hybrid
# dvxdyf_hybrid = 1j * KY * vxf_hybrid

# dvxdxf_standard = 1j * KX * vxf_standard
# dvxdyf_standard = 1j * KY * vxf_standard
# dvydxf_hybrid = 1j * KX * vyf_hybrid
# dvydyf_hybrid = 1j * KY * vyf_hybrid

# dvydxf_standard = 1j * KX * vyf_standard
# dvydyf_standard = 1j * KY * vyf_standard

# # Vorticity in Fourier space
# omegaf_hybrid = dvydxf_hybrid - dvxdyf_hybrid
# omegaf_standard = dvydxf_standard - dvxdyf_standard

# # Transform back to real space
# omega_hybrid = torch.real(torch.fft.ifft2(omegaf_hybrid))
# omega_standard = torch.real(torch.fft.ifft2(omegaf_standard))

# # Compute relative error
# rel_error_hybrid = torch.linalg.norm(omega_hybrid - omega) / torch.linalg.norm(omega)
# rel_error_standard = torch.linalg.norm(omega_standard - omega) / torch.linalg.norm(omega)
# print(f"Vorticity relative error hybrid MPS: {rel_error_hybrid.item():.6e}")
# print(f"Vorticity relative error standard MPS: {rel_error_standard.item():.6e}")

# # Divergence
# divf_hybrid = 1j * KX * vxf_hybrid + 1j * KY * vyf_hybrid
# divergence_hybrid = torch.real(torch.fft.ifft2(divf_hybrid))
# print(f"Max divergence hybrid MPS: {torch.max(divergence_hybrid).item()}")

# divf_standard = 1j * KX * vxf_standard + 1j * KY * vyf_standard
# divergence_standard = torch.real(torch.fft.ifft2(divf_standard))
# print(f"Max divergence standard MPS: {torch.max(divergence_standard).item()}")

# # Plot (convert to numpy for matplotlib)
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# im1 = axes[0].imshow(divergence_hybrid.cpu().numpy(), vmin=-10, vmax=10, cmap='RdBu')
# axes[0].set_title('Divergence Field (Hybrid MPS)')
# fig.colorbar(im1, ax=axes[0], label='Divergence')

# im2 = axes[1].imshow(divergence_standard.cpu().numpy(), vmin=-10, vmax=10, cmap='RdBu')
# axes[1].set_title('Divergence Field (Standard MPS)')
# fig.colorbar(im2, ax=axes[1], label='Divergence')

# plt.suptitle(f'Divergence for BD = {BD_MAX}', fontsize=16, y=1.02)
# plt.tight_layout()
# plt.show()


# # %% Visualizzo vx (original, hybrid, standard)
# field = "vx"
# vx32 = np.fromfile(f'{cfg.DATA_DIR}/{field}_real.000{config}', dtype = np.float32)
# vx32 = vx32[1:-1].reshape(4096,-1)
# vx = vx32.view(np.float64)
# vx = vx[:,:-2]
# vx = torch.from_numpy(vx)

# fig, axes = plt.subplots(1, 3, figsize=(18, 6))
# axes[0].imshow(vx, vmin = -2, vmax = 2,  cmap=cmap)
# axes[0].set_title(f"{field} Original")
# axes[1].imshow(vx_hybrid.detach().numpy(), vmin = -2, vmax = 2,  cmap=cmap)
# axes[1].set_title(f"{field} Hybrid MPS (N={N}, BD_equiv={BD_MAX})\nCR: {vx_hybrid_mps_compression}")
# axes[2].imshow(vx_standard.detach().numpy(), vmin = -2, vmax = 2,  cmap=cmap)
# axes[2].set_title(f"{field} Standard MPS (BD={BD_MAX})\nCR: {vx_standard_mps_compression}")
# plt.tight_layout()
# plt.show()



# # %% PDF of the increments for vx
# # Separation distance
# r_list = [1, 2]
# for l in range(11):
#     r_list.append(r_list[-1] * 2)

# vx_list = []
# vx_hybrid_list = []
# vx_standard_list = []

# vx_list.append(vx)
# vx_hybrid_list.append(vx_hybrid)
# vx_standard_list.append(vx_standard)

# t_hist = []
# t_hist_hybrid = []
# t_hist_standard = []
# bins = torch.arange(-20, 20, 0.1)

# for r in r_list:
#     all_dt = []
#     all_dt_hybrid = []
#     all_dt_standard = []
    
#     for vx in vx_list:
#         dvx = compute_increments(vx, axis=1, r=r, stride=downsample).ravel()
#         all_dt.append(dvx)

#     for vx_hybrid in vx_hybrid_list:
#         dvx_hybrid = compute_increments(vx_hybrid, axis=1, r=r, stride=downsample).ravel()
#         all_dt_hybrid.append(dvx_hybrid)

#     for vx_standard in vx_standard_list:
#         dvx_standard = compute_increments(vx_standard, axis=1, r=r, stride=downsample).ravel()
#         all_dt_standard.append(dvx_standard)
    
#     # Concatenate all configurations
#     all_dt = torch.cat(all_dt)
#     all_dt_hybrid = torch.cat(all_dt_hybrid)
#     all_dt_standard = torch.cat(all_dt_standard)
    
#     # --- Normalize by std (over all samples!)
#     all_dt_norm = all_dt / torch.std(all_dt)
#     all_dt_hybrid_norm = all_dt_hybrid / torch.std(all_dt_hybrid)
#     all_dt_standard_norm = all_dt_standard / torch.std(all_dt_standard)
    
#     # --- Compute PDFs
#     t_centers, t_pdf = compute_pdf(all_dt_norm, bins=bins.to(dtype=torch.float64))
#     t_centers, t_pdf_hybrid = compute_pdf(all_dt_hybrid_norm, bins=bins.to(dtype=torch.float64))
#     t_centers, t_pdf_standard = compute_pdf(all_dt_standard_norm, bins=bins.to(dtype=torch.float64))
    
#     t_hist.append(t_pdf)
#     t_hist_hybrid.append(t_pdf_hybrid)
#     t_hist_standard.append(t_pdf_standard)

# # --- Plot scalar PDF
# plt.figure(figsize=(8, 5))
# # Choose values of r to plot
# for r in [16]:
#     idx_r = r_list.index(r)
#     plt.semilogy(bins[:-1], t_hist[idx_r], label=r'$\ell_n = %.3f$'%(r/4096 * 2 * torch.pi)+', original')
#     plt.semilogy(bins[:-1], t_hist_hybrid[idx_r], label=r'$\ell_n = %.3f$'%(r/4096 * 2 * torch.pi)+f', hybrid, N = {N}')
#     plt.semilogy(bins[:-1], t_hist_standard[idx_r], label=r'$\ell_n = %.3f$'%(r/4096 * 2 * torch.pi)+', standard')

# plt.title(f'vx PDF for BD = {BD_MAX}')
# plt.xlabel('Normalized vx increment')
# plt.ylabel('PDF')
# plt.legend()
# plt.grid(True, which='both', linestyle='--', alpha=0.5)
# plt.tight_layout()
# plt.xlim(-20.0,20.0)
# plt.show()
# %%
