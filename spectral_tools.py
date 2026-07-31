"""
spectral_tools.py (simplified)

Assumptions
-----------
- The input field u is always a torch tensor of shape [3, 1024, 1024, 1024]
  (vector components first), real-valued float32/float64.
- Periodic boundary conditions are assumed.
- All routines run on CPU by default. Matplotlib uses non-interactive backend
  only in the calling script; these functions just save figures when asked.

Exposed functions
-----------------
1) total_energy = plot_energy_spectrum(u, out_png="energy_spectrum.png",
                                       inertial_range=(8,60), normalize_first_shell=True)
   - Computes the isotropic energy spectrum E(k) (shell-averaged).
   - If normalize_first_shell=True, normalizes the spectrum so that E(k=1)=1.
   - Fits and overlays a Kolmogorov k^{-5/3} line on the chosen inertial range.
   - Saves the plot to out_png and returns the TOTAL kinetic energy of the snapshot.

2) plot_increments_stats(u, out_dir=".", separations=(1,2,4,8,16,32), nbins=201)
   - Computes PDFs and flatness of longitudinal velocity increments:
     For component i in {0,1,2}, take increments along its own axis:
       du_i(r) = u_i(x + r e_i) - u_i(x), periodic wrap.
     Then average PDFs/flatness across i=0,1,2.
   - Saves flatness vs r as "<out_dir>/flatness_vs_r.png".
   - Also saves a PDF plot for the smallest and largest r:
       "<out_dir>/pdf_du_r<rmin>.png" and "<out_dir>/pdf_du_r<rmax>.png".

3) u_trunc, modes_zeroed_per_comp, k_cut = galerkin_truncate_by_M(u, M_per_comp, rng_seed=0)
   - Finds a (nearly) isotropic set of rFFT modes that keeps exactly M_per_comp scalar
     parameters per component (using rFFT weighting along the last axis). It keeps
     all |k| < k_boundary and randomly fills the |k| = k_boundary shell to hit
     exactly M_per_comp. The SAME mask is applied to all 3 components.
   - Returns the truncated field in REAL space (same shape as input),
     the number of (weighted) modes zeroed per component, and the boundary k_cut used.

+40,4) u_trunc, M_actual, k_cut = galerkin_truncate_2d(u, M_target)
   - Truncates a 2D scalar field (u: [Ny, Nx] or [C, Ny, Nx]) to a k_max such that
     the number of retained parameters is closest to M_target.
   - Returns the truncated field, the actual number of parameters kept, and the k_cut.

5) u_trunc = wavelet_truncate_2d(u, M_target)
   - Truncates a 2D scalar field by keeping only the top M_target coefficients
     of the 2D Haar Wavelet Transform.

6) k_vals, E_k = compute_isotropic_spectrum_2d(u)
   - Computes the shell-averaged isotropic energy spectrum for a 2D scalar field.
"""

from typing import Tuple, Sequence

import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt


# ------------------ Low-level helpers ------------------

def _assert_shape(u: torch.Tensor):
    if not (isinstance(u, torch.Tensor) and u.ndim == 4 and u.shape[0] == 3):
        raise ValueError(f"u must be [3,N,N,N], got {tuple(u.shape)}")
    if not (u.shape[1] == u.shape[2] == u.shape[3]):
        raise ValueError("spatial dims must be cubic (N×N×N)")


def _rfft_weights_last_axis(Nz: int, device: torch.device) -> torch.Tensor:
    """Weights for rFFT along last axis: 2 for interior freq bins, 1 for 0 and (if even) Nyquist."""
    w = torch.ones(Nz//2 + 1, device=device, dtype=torch.float64)
    if Nz % 2 == 0 and Nz > 0:
        if w.numel() > 2:
            w[1:-1] = 2.0
    else:
        w[1:] = 2.0
    return w

def _k_grids(Nx: int, Ny: int, Nz: int, device: torch.device):
    """Return integer-like kx, ky, kz grids (scaled by N so shells are integer-ish)."""
    kx = torch.fft.fftfreq(Nx, d=1.0).to(device) * Nx
    ky = torch.fft.fftfreq(Ny, d=1.0).to(device) * Ny
    kz = torch.fft.rfftfreq(Nz, d=1.0).to(device) * Nz
    return kx, ky, kz

def _k_grids_2d(Ny: int, Nx: int, device: torch.device):
    """Return integer-ish ky, kx grids for rFFT2."""
    ky = torch.fft.fftfreq(Ny, d=1.0).to(device) * Ny
    kx = torch.fft.rfftfreq(Nx, d=1.0).to(device) * Nx
    return ky, kx

def _isotropic_spectrum_from_uk(uk: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute shell-averaged spectrum from uk (shape [3,Nx,Ny,Nz//2+1], complex).
    Returns (k_vals, E_k) on CPU tensors.
    """
    device = uk.device
    _, Nx, Ny, Nzr = uk.shape
    Nz = (Nzr - 1) * 2

    # Energy density per spectral point (sum over 3 components)
    e_k = 0.5 * (uk.real**2 + uk.imag**2).sum(dim=0).to(torch.float64)  # (Nx,Ny,Nzr)
    # rFFT weights along last axis
    wlast = _rfft_weights_last_axis(Nz, device=device).view(1,1,-1)
    e_k = e_k * wlast

    # Shell indices
    kx, ky, kz = _k_grids(Nx, Ny, Nz, device=device)
    KX = kx.view(-1,1,1); KY = ky.view(1,-1,1); KZ = kz.view(1,1,-1)
    shell = torch.floor(torch.sqrt(KX*KX + KY*KY + KZ*KZ)).to(torch.int64)

    shell_cpu = shell.flatten().to('cpu')
    e_cpu = e_k.flatten().to('cpu')
    kmax = int(shell_cpu.max().item())
    E = torch.bincount(shell_cpu, weights=e_cpu, minlength=kmax+1).to(torch.float64)
    k_vals = torch.arange(kmax+1, dtype=torch.float64)
    return k_vals, E

def _kolmogorov_fit(k: torch.Tensor, E: torch.Tensor, kmin: int, kmax: int) -> Tuple[torch.Tensor, float]:
    """Fit E ~ C k^{-5/3} on [kmin,kmax]. Return E_fit over all k and C."""
    E = E.clone()
    mask = (k >= kmin) & (k <= kmax) & (E > 0) & torch.isfinite(E)
    if mask.sum() < 2:
        # not enough points
        ksafe = k.clone(); ksafe[0] = 1
        Efit = (ksafe**(-5.0/3.0))
        Efit[0] = float('nan')
        return Efit, 1.0
    logk = torch.log(k[mask])
    logE = torch.log(E[mask])
    slope = -5.0/3.0
    a = (logE - slope*logk).mean()
    C = float(torch.exp(a).item())
    ksafe = k.clone(); ksafe[0] = 1
    Efit = C * (ksafe**(-5.0/3.0))
    Efit[0] = float('nan')
    return Efit, C

# ------------------ Public functions ------------------

def plot_energy_spectrum(u: torch.Tensor, out_png: str = "energy_spectrum.png",
                         inertial_range: Tuple[int,int] = (8,60),
                         normalize_first_shell: bool = True) -> float:
    """
    Compute isotropic energy spectrum E(k), normalize so E(1)=1 if requested,
    fit Kolmogorov -5/3 on inertial_range, save plot, and return TOTAL energy.
    """
    _assert_shape(u)
    device = u.device

    # rFFT over last 3 dims (vectorized over components)
    uk = torch.fft.rfftn(u, dim=(-3,-2,-1), norm='ortho')

    # total energy via Parseval (using same e_k as spectrum)
    k, E = _isotropic_spectrum_from_uk(uk)
    total_energy = float(E.sum().item())

    # normalization
    E_plot = E.clone()
    if normalize_first_shell and E_plot.numel() > 2 and E_plot[1] > 0:
        E_plot = E_plot / E_plot[1]

    # fit and plot
    kmin, kmax = inertial_range
    Efit, C = _kolmogorov_fit(k, E_plot, kmin, kmax)

    plt.figure(figsize=(8,5))
    # skip k=0
    kn = k[1:].numpy()
    En = E_plot[1:].numpy()
    Efn = Efit[1:].numpy()
    plt.loglog(kn, En, label="E(k)")
    plt.loglog(kn, Efn, linestyle='--', label=r"fit $\propto k^{-5/3}$")
    if normalize_first_shell:
        plt.axhline(1.0, color='gray', linestyle=':', linewidth=1.0, label="E(1)=1 norm")
    plt.xlabel("k (shell index)")
    plt.ylabel("E(k)" + (" / E(1)" if normalize_first_shell else ""))
    plt.title(f"Isotropic spectrum; -5/3 fit on [{kmin},{kmax}]")
    plt.grid(True, which='both', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    return total_energy

def plot_increments_stats(
    u: torch.Tensor,
    out_dir: str = ".",
    separations: Sequence[int] = (1, 2, 4, 8, 16, 32),
    nbins: int = 201,
    pdf_out_png: str = "pdf_du_all_r.png",
    pdf_title: str = "PDF of longitudinal increments (avg over components)",
) -> None:
    """
    Compute longitudinal velocity increments per component and aggregate stats:
      du_x(r) = u_x(x+r, y, z) - u_x(x,y,z)
      du_y(r) = u_y(x, y+r, z) - u_y(x,y,z)
      du_z(r) = u_z(x, y, z+r) - u_z(x,y,z)

    Saves:
      - flatness_vs_r.png  (in out_dir)
      - one combined PDF figure with ALL separations overlaid (pdf_out_png)

    Args:
        u: torch tensor [3, N, N, N] (assumed periodic).
        out_dir: directory to save outputs.
        separations: list/tuple of integer grid separations r.
        nbins: number of histogram bins + 1 for edges (use nbins-1 bins).
        pdf_out_png: filename (or path) for the combined PDF plot.
        pdf_title: title to put on the combined PDF plot.
    """
    _assert_shape(u)
    os.makedirs(out_dir, exist_ok=True)

    device = u.device
    u = u.to(torch.float64)
    rs = list(separations)

    flatness_vals = []
    stds = []

    # ---------- PASS 1: moments (to set a common PDF range) ----------
    for r in rs:
        # increments for each component along its own axis
        dux = u[0].roll(shifts=r, dims=0) - u[0]
        duy = u[1].roll(shifts=r, dims=1) - u[1]
        duz = u[2].roll(shifts=r, dims=2) - u[2]

        du_all = torch.cat([dux.flatten(), duy.flatten(), duz.flatten()])
        m2 = torch.mean(du_all**2)
        m4 = torch.mean(du_all**4)
        flat = float((m4 / (m2*m2 + 1e-30)).item())
        flatness_vals.append(flat)

        stds.append(float(torch.sqrt(m2).item()))

    # Save flatness vs r
    plt.figure(figsize=(7, 4.5))
    plt.plot(rs, flatness_vals, marker="o")
    plt.xscale("log", base=2)
    plt.xlabel("Separation r (grid points)")
    plt.ylabel("Flatness = <du^4>/<du^2>^2 (avg over components)")
    plt.title("Flatness of longitudinal velocity increments")
    plt.grid(True, which="both", alpha=0.4)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "flatness_vs_r.png"), dpi=200)
    plt.close()

    # ---------- PASS 2: PDFs with a common bin range ----------
    # Use a robust shared range: ±6 * max(std over r)
    vmax = max(1e-9, 6.0 * max(stds))
    bins = nbins - 1
    edges = torch.linspace(-vmax, vmax, steps=bins + 1, device=device)
    dx = (2 * vmax) / bins
    x_for_plot = (edges[:-1] + edges[1:]) / 2  # bin centers

    plt.figure(figsize=(8, 5))
    for r in rs:
        dux = u[0].roll(shifts=r, dims=0) - u[0]
        duy = u[1].roll(shifts=r, dims=1) - u[1]
        duz = u[2].roll(shifts=r, dims=2) - u[2]
        du_all = torch.cat([dux.flatten(), duy.flatten(), duz.flatten()])

        # counts over fixed range, then convert to PDF
        counts = torch.histc(du_all, bins=bins, min=-vmax, max=vmax)
        pdf = counts / counts.sum() / dx
        plt.plot(x_for_plot.cpu().numpy(), pdf.cpu().numpy(), label=f"r={r}")

    plt.xlabel("du")
    plt.ylabel("PDF")
    plt.title(pdf_title)
    plt.grid(True, alpha=0.4)
    plt.legend(title="separation")
    plt.tight_layout()
    out_path = pdf_out_png if os.path.isabs(pdf_out_png) else os.path.join(out_dir, pdf_out_png)
    plt.savefig(out_path, dpi=200)
    plt.close()


def _exact_M_mask(Nx: int, Ny: int, Nz: int, M_target: float, device: torch.device, rng_seed: int = 0):
    """
    Build boolean mask (Nx,Ny,Nz//2+1) that keeps exactly M_target weighted scalar modes (rFFT layout).
    Strategy: keep all |k| < k_boundary; partially fill |k| == k_boundary at random to reach exact M_target.
    Returns (mask, k_boundary).
    """
    torch.manual_seed(rng_seed)
    kx, ky, kz = _k_grids(Nx, Ny, Nz, device)
    KX = kx.view(-1,1,1); KY = ky.view(1,-1,1); KZ = kz.view(1,1,-1)
    kmag = torch.sqrt(KX*KX + KY*KY + KZ*KZ)
    wlast = _rfft_weights_last_axis(Nz, device=device).view(1,1,-1)
    w = torch.ones_like(kmag, dtype=torch.float64) * wlast

    kf = kmag.flatten()
    wf = w.flatten()
    order = torch.argsort(kf)
    k_sorted = kf[order]
    w_sorted = wf[order]
    cume = torch.cumsum(w_sorted, dim=0)

    M_target = float(M_target)
    idx = torch.searchsorted(cume, torch.tensor(M_target, device=device))
    idx = min(int(idx.item()), cume.numel()-1)
    k_boundary = float(k_sorted[idx].item())

    mask_flat = torch.zeros_like(kf, dtype=torch.bool)
    below = (kf < k_boundary)
    mask_flat[below] = True
    M_in = float(wf[below].sum().item())
    need = M_target - M_in
    if need > 0:
        on = (kf == k_boundary)
        b_idx = torch.nonzero(on, as_tuple=False).view(-1)
        perm = torch.randperm(b_idx.numel(), device=device)
        b_idx = b_idx[perm]
        s = 0.0
        chosen = []
        for j in b_idx:
            wj = float(wf[j].item())
            if s + wj <= need + 1e-9:
                chosen.append(int(j))
                s += wj
            if abs(s - need) <= 1e-9 or s > need:
                break
        mask_flat[chosen] = True

    return mask_flat.view(kmag.shape), k_boundary

def galerkin_truncate_by_M(u: torch.Tensor, M_per_comp: float, rng_seed: int = 0):
    """
    Keep exactly M_per_comp weighted scalar rFFT modes PER component by:
      - building one mask (Nx,Ny,Nz//2+1) via exact-M selection
      - applying the same mask to the rFFT of each component
      - inverse rFFT back to real space

    Returns:
      u_trunc (real tensor, same shape as input),
      modes_zeroed_per_comp (float, weighted count),
      k_cut (float, boundary magnitude used).
    """
    _assert_shape(u)
    device = u.device
    _, Nx, Ny, Nz = u.shape

    uk = torch.fft.rfftn(u, dim=(-3,-2,-1), norm='ortho')  # (3,Nx,Ny,Nz//2+1)
    mask, k_cut = _exact_M_mask(Nx, Ny, Nz, float(M_per_comp), device, rng_seed=rng_seed)

    uk_masked = uk * mask.view(1, Nx, Ny, Nz//2+1)

    # Weighted mode counts per component
    wlast = _rfft_weights_last_axis(Nz, device=device).view(1,1,-1)
    modes_total = float((torch.ones(Nx,Ny,Nz//2+1, dtype=torch.float64, device=device) * wlast).sum().item())
    modes_kept = float((mask.to(torch.float64) * wlast).sum().item())
    modes_zeroed = modes_total - modes_kept

    u_trunc = torch.fft.irfftn(uk_masked, s=(Nx,Ny,Nz), dim=(-3,-2,-1), norm='ortho').real
    return u_trunc, modes_zeroed, k_cut


def galerkin_truncate_2d(u: torch.Tensor, M_target: float):
    """
    Truncate a 2D field (u: [Ny, Nx] or [C, Ny, Nx]) to an isotropic k_max such that
    the number of retained parameters is closest to M_target.

    Parameters:
    -----------
    u : torch.Tensor
        Input 2D field.
    M_target : float
        Target number of real parameters to keep.

    Returns:
    --------
    u_trunc : torch.Tensor
        Truncated field in real space.
    M_actual : float
        Actual number of real parameters kept.
    k_cut : float
        The wave-number magnitude used for truncation.
    """
    device = u.device
    if u.ndim == 2:
        Ny, Nx = u.shape
        u_in = u.unsqueeze(0)
    elif u.ndim == 3:
        _, Ny, Nx = u.shape
        u_in = u
    else:
        raise ValueError(f"Expected 2D or 3D tensor [C, Ny, Nx], got {u.ndim}D")

    # rFFT2
    uk = torch.fft.rfft2(u_in, norm='ortho')
    Nyr, Nxr = uk.shape[-2:]

    # Grid and weights
    ky, kx = _k_grids_2d(Ny, Nx, device=device)
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')
    kmag = torch.sqrt(KX**2 + KY**2)
    
    # Weighting for real parameters: 2 for interior rFFT modes, 1 for real boundaries
    wlast = _rfft_weights_last_axis(Nx, device=device).view(1, -1)
    # Sum of weights in a real N*N field is exactly N*N.
    # For isotropic truncation, we want to find k_max that collects M_target weights.
    w_all = (torch.ones_like(kmag, dtype=torch.float64) * wlast)

    # Find k_max for closest parameter count
    kmag_f = kmag.flatten()
    w_f = w_all.flatten()
    
    # Sort by k magnitude
    unique_k, inverse_indices = torch.unique(kmag_f, return_inverse=True)
    k_weights = torch.bincount(inverse_indices, weights=w_f, minlength=unique_k.numel())
    k_cumsum = torch.cumsum(k_weights, dim=0)
    
    # Finding index of closest M_target
    idx = torch.argmin(torch.abs(k_cumsum - M_target))
    k_cut = float(unique_k[idx].item())
    M_actual = float(k_cumsum[idx].item())

    # Apply mask
    mask = (kmag <= k_cut)
    uk_masked = uk * mask.view(1, Ny, Nxr)

    # Inverse rFFT
    u_trunc = torch.fft.irfft2(uk_masked, s=(Ny, Nx), norm='ortho').real

    if u.ndim == 2:
        u_trunc = u_trunc.squeeze(0)

    return u_trunc, M_actual, k_cut


def haar_dwt_2d(x: torch.Tensor):
    """Simple 1-level 2D Haar DWT using PyTorch."""
    # x: [C, H, W]
    x00 = x[:, 0::2, 0::2]
    x01 = x[:, 0::2, 1::2]
    x10 = x[:, 1::2, 0::2]
    x11 = x[:, 1::2, 1::2]
    
    # LL, LH, HL, HH
    ll = 0.5 * (x00 + x01 + x10 + x11)
    lh = 0.5 * (x00 - x01 + x10 - x11)
    hl = 0.5 * (x00 + x01 - x10 - x11)
    hh = 0.5 * (x00 - x01 - x10 + x11)
    return ll, (lh, hl, hh)

def haar_idwt_2d(ll: torch.Tensor, details: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]):
    """Simple 1-level 2D Haar IDWT using PyTorch."""
    lh, hl, hh = details
    # Inverse Haar
    x00 = 0.5 * (ll + lh + hl + hh)
    x01 = 0.5 * (ll - lh + hl - hh)
    x10 = 0.5 * (ll + lh - hl - hh)
    x11 = 0.5 * (ll - lh - hl + hh)
    
    C, H, W = ll.shape
    out = torch.empty((C, 2 * H, 2 * W), device=ll.device, dtype=ll.dtype)
    out[:, 0::2, 0::2] = x00
    out[:, 0::2, 1::2] = x01
    out[:, 1::2, 0::2] = x10
    out[:, 1::2, 1::2] = x11
    return out

def wavelet_truncate_2d(u: torch.Tensor, M_target: float):
    """
    Truncate a 2D field by keeping top M_target Haar wavelet coefficients.
    """
    device = u.device
    if u.ndim == 2:
        Ny, Nx = u.shape
        u_in = u.unsqueeze(0)
    else:
        _, Ny, Nx = u.shape
        u_in = u
        
    # Multi-level Haar decomposition until possible
    levels = 0
    temp_ny, temp_nx = Ny, Nx
    while temp_ny % 2 == 0 and temp_nx % 2 == 0 and temp_ny > 1 and temp_nx > 1:
        temp_ny //= 2
        temp_nx //= 2
        levels += 1
        
    if levels == 0:
        # Fallback if dims are not even
        return u
        
    coeffs = []
    current_ll = u_in
    all_details = []
    
    for l in range(levels):
        current_ll, (lh, hl, hh) = haar_dwt_2d(current_ll)
        all_details.append((lh, hl, hh))
        
    # Flat list of all coefficients to threshold
    flat_coeffs = [current_ll.flatten()]
    for d in all_details:
        for component in d:
            flat_coeffs.append(component.flatten())
            
    all_coeffs_flat = torch.cat(flat_coeffs)
    
    # Thresholding: Keep top M_target by magnitude
    # M_target might be per channel or total. Let's assume total parameters if multi-channel.
    # The user input implies dof = weight * 2^24 which is total parameters for NxN.
    M_target = int(round(float(M_target)))
    if M_target >= all_coeffs_flat.numel():
        return u
        
    # Get threshold
    mags = torch.abs(all_coeffs_flat)
    val, _ = torch.topk(mags, M_target)
    threshold = val[-1]
    
    # Apply thresholding
    def apply_thresh(t, thresh):
        mask = (torch.abs(t) >= thresh)
        # Note: if many have same magnitude exactly at threshold, we might keep slightly more.
        # But topk ensures we have at least M_target.
        return t * mask
    
    trunc_ll = apply_thresh(current_ll, threshold)
    trunc_details = []
    for d in all_details:
        trunc_details.append(tuple(apply_thresh(c, threshold) for c in d))
        
    # Reconstruction
    recon = trunc_ll
    for l in reversed(range(levels)):
        recon = haar_idwt_2d(recon, trunc_details[l])
        
    if u.ndim == 2:
        recon = recon.squeeze(0)

    return recon


def wavelet_truncate_2d_pywt(u: torch.Tensor, M_target: float,
                             wavelet: str = 'db4', mode: str = 'periodization'):
    """
    Truncate a 2D field by keeping the top-M wavelet coefficients (by magnitude),
    using PyWavelets so higher-order wavelets (db4, coif, bior, ...) are available.

    Mirrors `wavelet_truncate_2d` (the hand-rolled Haar version): the same global
    hard-threshold rule is applied to ALL coefficients, so the only thing that
    changes vs. the Haar baseline is the wavelet family. This keeps the
    DOF-matched comparison in the figure scripts apples-to-apples.

    Parameters
    ----------
    u        : real field, 2D torch tensor [Ny, Nx] (or [1, Ny, Nx]).
    M_target : target number of non-zero coefficients to keep.
    wavelet  : PyWavelets wavelet name, e.g. 'db4', 'bior4.4', 'coif2'.
    mode     : signal-extension mode. 'periodization' keeps the total coefficient
               count equal to the number of pixels (ideal for periodic,
               power-of-two fields and for an exact top-M count).
    """
    import pywt

    squeeze = False
    if u.ndim == 3:
        if u.shape[0] != 1:
            raise ValueError("wavelet_truncate_2d_pywt expects a single 2D field "
                             f"(got leading dim {u.shape[0]}).")
        u2d, squeeze = u[0], True
    else:
        u2d = u

    device, dtype = u.device, u.dtype
    arr = u2d.detach().cpu().numpy()

    # Full multilevel separable 2D DWT, flattened to a single coefficient array.
    coeffs = pywt.wavedec2(arr, wavelet=wavelet, mode=mode)
    arr_coeffs, slices = pywt.coeffs_to_array(coeffs)

    M_target = int(round(float(M_target)))
    if M_target >= arr_coeffs.size:
        return u  # nothing to truncate

    # Hard threshold: keep the M_target largest-magnitude coefficients.
    mags = np.abs(arr_coeffs).ravel()
    # M_target-th largest magnitude (>= keeps at least M_target; ties may keep a few more).
    threshold = np.partition(mags, mags.size - M_target)[mags.size - M_target]
    arr_coeffs = np.where(np.abs(arr_coeffs) >= threshold, arr_coeffs, 0.0)

    # Reconstruct.
    coeffs_t = pywt.array_to_coeffs(arr_coeffs, slices, output_format='wavedec2')
    recon = pywt.waverec2(coeffs_t, wavelet=wavelet, mode=mode)
    recon = recon[:arr.shape[0], :arr.shape[1]]  # crop any padding row/col

    out = torch.from_numpy(np.ascontiguousarray(recon)).to(device=device, dtype=dtype)
    if squeeze:
        out = out.unsqueeze(0)
    return out


def compute_isotropic_spectrum_2d(u: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute shell-averaged isotropic spectrum for a 2D scalar field.
    u: [Ny, Nx] or [C, Ny, Nx]
    Returns (k_vals, E_k)
    """
    device = u.device
    if u.ndim == 2:
        Ny, Nx = u.shape
        u_in = u.unsqueeze(0)
    else:
        C, Ny, Nx = u.shape
        u_in = u
        
    # rFFT2
    uk = torch.fft.rfft2(u_in, norm='ortho') # [C, Ny, Nxr]
    Nyr, Nxr = uk.shape[-2:]
    
    # Energy density: sum over channels
    e_k = 0.5 * (uk.real**2 + uk.imag**2).sum(dim=0).to(torch.float64) # [Ny, Nxr]
    
    # rFFT weights for last axis (Nx)
    wlast = _rfft_weights_last_axis(Nx, device=device).view(1, -1)
    e_k = e_k * wlast
    
    # K-magnitude grid
    ky, kx = _k_grids_2d(Ny, Nx, device=device)
    KY, KX = torch.meshgrid(ky, kx, indexing='ij')
    kmag = torch.sqrt(KX**2 + KY**2)
    
    # Binning by integer shell
    shell = torch.floor(kmag + 0.5).to(torch.int64) # Round to nearest integer shell
    
    shell_cpu = shell.flatten().to('cpu')
    e_cpu = e_k.flatten().to('cpu')
    kmax = int(shell_cpu.max().item())
    
    E = torch.bincount(shell_cpu, weights=e_cpu, minlength=kmax+1).to(torch.float64)
    k_vals = torch.arange(kmax+1, dtype=torch.float64)
    
    return k_vals, E

# ------------------ Example usage (cluster-ready) ------------------

if __name__ == "__main__":
    # This example *generates* a random [3,1024,1024,1024] tensor,
    # computes its total energy and spectrum (saved to PNG),
    # then truncates to only 100 weighted rFFT modes per component,
    # saves the truncated spectrum, and reports energy loss.
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=".", help="Where to save plots")
    parser.add_argument("--save-plots", action="store_true", help="Save spectrum and increment plots")
    parser.add_argument("--run-increments", action="store_true", help="Also compute flatness/PDF plots")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # WARNING: allocate only on a node with ample RAM.
    u = torch.randn(3, 1024, 1024, 1024, dtype=torch.float32)

    total_E_before = plot_energy_spectrum(
        u, out_png=os.path.join(args.outdir, "energy_spectrum.png"),
        inertial_range=(8,60), normalize_first_shell=True
    )
    print(f"Total energy (before truncation): {total_E_before:.6e}")

    u_trunc, zeroed, kcut = galerkin_truncate_by_M(u, M_per_comp=100, rng_seed=1)
    print(f"Galerkin truncation: k_cut={kcut:.3f}, zeroed modes per comp (weighted)={zeroed:.1f}")

    total_E_after = plot_energy_spectrum(
        u_trunc, out_png=os.path.join(args.outdir, "energy_spectrum_truncated.png"),
        inertial_range=(8,60), normalize_first_shell=True
    )
    print(f"Total energy (after truncation): {total_E_after:.6e}")
    lost = (total_E_before - total_E_after) / max(total_E_before, 1e-30)
    print(f"Energy lost fraction: {lost:.6%}")

    if args.run_increments and args.save_plots:
        plot_increments_stats(u, out_dir=args.outdir, separations=(1,2,4,8,16,32), nbins=201)
        print("Saved flatness and PDF plots.")
