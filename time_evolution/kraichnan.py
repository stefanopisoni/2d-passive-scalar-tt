"""Kraichnan passive scalar in the hybrid TT encoding.

Model
-----
    d_t theta + u . grad theta = kappa lap theta + f   on [0, 2pi)^2,

with u Gaussian, incompressible and white in time, and f white in time,

    <u_i(x,t) u_j(x',t')> = D_ij(x - x') delta(t - t'),
    D_ij(r) = sum_{q in H} w_q e_i(q) e_j(q) cos(q . r),
    <f(x,t) f(x',t')>     = Phi(x - x') delta(t - t'),
    Phi(r)  = sum_{k in F} phi_k cos(k . r),

H and F being sets of lattice wavevectors in a half-plane. The velocity weights
follow w_q ~ |q|^-(2+xi) for 1 <= |q| <= q_max, so the time-integrated velocity
structure function grows as r^xi below the largest scale: xi = 4/3 gives the
Obukhov-Corrsin scalar spectrum k^-5/3, and xi = 2 (velocity confined to the
largest scales) the Batchelor regime k^-1.

Renewal discretisation
----------------------
Time is cut into intervals of length tau. In each interval the velocity is one
mode, u = a_q e(q) sin(q.x + phi), held fixed, with q drawn with probability
p_q and phi uniform; the scalar is advanced through the interval by the
deterministic solver, and a forcing increment c_k cos(k.x + phi') is added at the
end of it. With

    a_q^2 = 2 w_q / (tau p_q),      c_k^2 = 2 tau phi_k / p_k,

the covariance of the velocity integrated over an interval is D tau and that of
the forcing increment Phi tau. In the white-noise limit the scalar sees the
velocity only through D_ij, so one mode per interval suffices: Gaussianity is
reached by accumulating many independent intervals, and the exact flow over an
interval differs from the white-noise generator at relative order Ku^2.

Choosing p_q ~ w_q |s(q)|^2 makes the Kubo number of an interval,
Ku = a_q tau |s(q)|, the same for every mode: Ku^2 = 2 tau Z with
Z = sum_q w_q |s(q)|^2. Ku is the one parameter measuring the distance from the
white-noise limit; tau follows from it.

Discrete construction
---------------------
A plane wave cos(q.x + phi) factorises over the binary digits of x and y, so it
is an exact rank-2 tensor train (one 2x2 rotation per leg) and an exact rank-2
hybrid TT; :func:`plane_wave_hybrid` builds it without forming the dense field.
The velocity is the discrete curl of psi = b cos(q.x + phi) with the solver's
centred stencils, u = (D_y psi, -D_x psi) = b (-s_y, s_x) sin(q.x + phi), where
s(q) = sin(q dx)/dx is the centred-difference symbol. It is divergence-free on
the grid exactly, and e(q) = (-s_y, s_x)/|s|.

Lattice benchmark
-----------------
For the semi-discrete scheme (centred skew-form transport, 5-point Laplacian) in
the white-noise limit, the spectrum C(k) = E|theta_hat_k|^2, with
theta_hat = fft2(theta)/n^2, obeys exactly, aliasing included:

    dC(k)/dt = sum_q W(q) g_q(k, k-q)^2 [C(k-q) - C(k)]
               - 2 kappa lam(k) C(k) + Phi_hat(k),

    W(+-q) = w_q / 8,   g_q(k, k') = e(q) . (s(k) + s(k')),   Phi_hat(+-k) = phi_k / 2,

lam(k) the symbol of the 5-point Laplacian. The derivation: over one interval
theta -> exp(-tau A) theta with A the anti-Hermitian transport operator, so the
second moment M changes by (tau^2/2) E[A,[A,M]] + O(tau^4), and the phase
average keeps M diagonal in Fourier space. :func:`lattice_spectrum` solves the
stationary equation; tests/test_kraichnan.py checks the operator against the
brute-force double commutator on a 16x16 grid.
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

import hybrid_ops as ho

# forcing shell, centred on k_f = 3
KF_LO, KF_HI, KF = 2.5, 3.5, 3.0


def half_plane(qmax: float, qmin: float = 1.0):
    """Integer wavevectors (qx, qy) with qmin <= |q| <= qmax, one of each +-q pair."""
    m = int(np.floor(qmax))
    out = []
    for qy in range(0, m + 1):
        for qx in range(-m, m + 1):
            if qy == 0 and qx <= 0:
                continue
            r = np.hypot(qx, qy)
            if qmin - 1e-12 <= r <= qmax + 1e-12:
                out.append((qx, qy))
    return np.array(out, dtype=np.int64)


def symbol(q: np.ndarray, dx: float) -> np.ndarray:
    """Centred-difference symbol s(q) = sin(q dx)/dx, componentwise."""
    return np.sin(np.asarray(q, dtype=float) * dx) / dx


@dataclass
class Kraichnan:
    """Velocity and forcing statistics, and the renewal sampler for them.

    ``xi`` sets the velocity spectrum, ``qmax`` its cutoff, ``ku`` the Kubo
    number of an interval. The amplitude is normalised so that the deformation
    rate at the forcing scale, Gamma(k_f) = 1/2 sum_q w_q min(|s(q)|^2, k_f^2),
    is one: modes larger than the forcing scale strain it, smaller ones diffuse
    it, and the time unit is the forcing-scale turnover for every xi.
    """

    L: int
    xi: float
    qmax: float
    ku: float = 0.5
    injection: float = 1.0       # Phi(0): variance injection rate d<theta^2>/dt

    def __post_init__(self):
        self.n = 2 ** self.L
        self.dx = 2 * np.pi / self.n
        q = half_plane(self.qmax)
        if np.any(2 * np.abs(q) >= self.n):
            raise ValueError("velocity modes must stay below the Nyquist wavenumber")
        s = symbol(q, self.dx)
        s2 = (s ** 2).sum(1)
        w = np.hypot(q[:, 0], q[:, 1]) ** (-(2.0 + self.xi))
        w /= 0.5 * np.sum(w * np.minimum(s2, KF ** 2))
        self.q, self.s, self.w = q, s, w
        self.e = np.stack([-s[:, 1], s[:, 0]], 1) / np.sqrt(s2)[:, None]
        Z = float(np.sum(w * s2))
        self.p = w * s2 / Z
        self._cdf = np.cumsum(self.p)
        self.tau = self.ku ** 2 / (2.0 * Z)
        self.amp = self.ku / (self.tau * np.sqrt(s2))      # a_q, velocity amplitude
        self.psi_amp = self.amp / np.sqrt(s2)              # b_q, streamfunction amplitude

        f = half_plane(KF_HI, KF_LO)
        self.kf = f
        self.phi = np.full(len(f), self.injection / len(f))
        self.pf = np.full(len(f), 1.0 / len(f))
        self.famp = np.sqrt(2.0 * self.tau * self.phi / self.pf)

    # ------------------------------------------------------------------ stats
    def gamma(self, k: float) -> float:
        """Deformation rate at wavenumber k (both straining and eddy-diffusing modes)."""
        return float(0.5 * np.sum(self.w * np.minimum((self.s ** 2).sum(1), k * k)))

    def eddy_diffusivity(self) -> float:
        """K with D_ij(0) = 2 K delta_ij for an isotropic set: the Ito drift is K lap."""
        return float(0.25 * self.w.sum())

    def default_kappa(self) -> float:
        """kappa placing the diffusive cutoff at a fixed fraction of the grid.

        Batchelor regime (velocity confined to the large scales): the Batchelor
        wavenumber sqrt(Gamma/kappa) is put at n/18, as in the cellular ladder
        (5e-3 at 256^2, ell_B/dx ~ 2.9). Rough velocity: the turbulent and
        molecular rates are balanced at k_d = n/16, Gamma(k_d) = kappa k_d^2.
        """
        if self.qmax < KF_LO:
            return 5e-3 * 4.0 ** (8 - self.L)
        kd = self.n / 16
        return self.gamma(kd) / kd ** 2

    # ---------------------------------------------------------------- sampling
    def draw(self, rng: np.random.Generator):
        """One interval: (velocity mode index, phase, forcing mode index, phase)."""
        i = min(int(np.searchsorted(self._cdf, rng.random(), side="right")),
                len(self.q) - 1)
        j = int(rng.integers(len(self.kf)))
        return i, float(rng.uniform(0, 2 * np.pi)), j, float(rng.uniform(0, 2 * np.pi))

    def velocity_dense(self, i: int, ph: float, XX, YY):
        """(ux, uy) = (D_y psi, -D_x psi) for psi = b cos(q.x + ph), evaluated analytically."""
        (qx, qy), (sx, sy), b = self.q[i], self.s[i], self.psi_amp[i]
        sn = np.sin(qx * XX + qy * YY + ph)
        return -b * sy * sn, b * sx * sn

    def velocity_hybrid(self, i: int, ph: float, N: int):
        """The same velocity as exact rank-2 hybrid TTs (sin = cos shifted by -pi/2)."""
        (qx, qy), (sx, sy), b = self.q[i], self.s[i], self.psi_amp[i]
        return (plane_wave_hybrid(self.L, N, qx, qy, ph - np.pi / 2, -b * sy),
                plane_wave_hybrid(self.L, N, qx, qy, ph - np.pi / 2, b * sx))

    def forcing_dense(self, j: int, ph: float, XX, YY):
        kx, ky = self.kf[j]
        return self.famp[j] * np.cos(kx * XX + ky * YY + ph)

    def forcing_hybrid(self, j: int, ph: float, N: int):
        kx, ky = self.kf[j]
        return plane_wave_hybrid(self.L, N, kx, ky, ph, self.famp[j])

    def describe(self) -> dict:
        return dict(xi=self.xi, qmax=self.qmax, ku=self.ku, tau=self.tau,
                    n_vel_modes=len(self.q), n_forcing_modes=len(self.kf),
                    eddy_diffusivity=self.eddy_diffusivity(),
                    gamma_kf=self.gamma(KF), gamma_total=self.gamma(np.inf),
                    amp_max=float(self.amp.max()), injection=self.injection)


# --------------------------------------------------------------------------
# exact rank-2 plane waves in hybrid form
# --------------------------------------------------------------------------
def leg_angles(L: int, qx: float, qy: float) -> np.ndarray:
    """Phase carried by each interleaved leg's bit: even legs rows (y), odd columns (x).

    Leg 2l holds bit l (MSB first) of the row index, worth 2^(L-1-l) cells.
    """
    dx = 2 * np.pi / 2 ** L
    a = np.empty(2 * L)
    for l in range(L):
        w = dx * 2 ** (L - 1 - l)
        a[2 * l] = qy * w
        a[2 * l + 1] = qx * w
    return a


def plane_wave_hybrid(L: int, N: int, qx: float, qy: float, phase: float,
                      amp: float = 1.0):
    """amp * cos(qx x + qy y + phase) as an exact rank-2 hybrid TT.

    The state (cos t, sin t) is carried along the chain and rotated by each
    leg's angle when its bit is set; the last core reads off the cosine. No dense
    field is formed, so the cost is O(2^N + L) whatever the grid.
    """
    ho._check_LN(L, N)
    a = leg_angles(L, qx, qy)
    dt = torch.get_default_dtype()
    X = np.arange(2 ** N)
    bits = (X[:, None] >> (N - 1 - np.arange(N))[None, :]) & 1 if N else np.zeros((1, 0))
    t = phase + bits @ a[:N]
    head = np.stack([np.cos(t), np.sin(t)], 1)[None]          # (1, 2^N, 2)
    cores = [torch.from_numpy(head).to(dt)]
    for j in range(N, 2 * L):
        c, s = np.cos(a[j]), np.sin(a[j])
        T = np.zeros((2, 2, 2))
        T[:, 0, :] = np.eye(2)
        T[:, 1, :] = [[c, s], [-s, c]]
        cores.append(torch.from_numpy(T).to(dt))
    last = cores[-1][:, :, :1] * amp                          # cosine component
    cores[-1] = last
    return ho.from_tensors(cores)


# --------------------------------------------------------------------------
# lattice-exact second moment
# --------------------------------------------------------------------------
def _lattice(n: int, dx: float):
    k = np.fft.fftfreq(n, d=1.0 / n)
    KY, KX = np.meshgrid(k, k, indexing="ij")                 # rows y, cols x
    sx, sy = np.sin(KX * dx) / dx, np.sin(KY * dx) / dx
    lam = 4.0 * (np.sin(KX * dx / 2) ** 2 + np.sin(KY * dx / 2) ** 2) / dx ** 2
    return KX, KY, sx, sy, lam


class LatticeOperator:
    """Right-hand side of the closed equation for C(k), with C as an n x n array
    in numpy fft ordering (rows k_y, columns k_x)."""

    def __init__(self, model: Kraichnan, kappa: float):
        n, dx = model.n, model.dx
        self.n, self.kappa = n, kappa
        _, _, self.sx, self.sy, self.lam = _lattice(n, dx)
        W = {c: np.zeros((n, n)) for c in ("xx", "xy", "yy")}
        for (qx, qy), w, (ex, ey) in zip(model.q, model.w, model.e):
            for sg in (1, -1):
                r, c = (sg * qy) % n, (sg * qx) % n
                W["xx"][r, c] += w / 8 * ex * ex
                W["xy"][r, c] += w / 8 * ex * ey
                W["yy"][r, c] += w / 8 * ey * ey
        self.FW = {c: np.fft.fft2(v) for c, v in W.items()}
        self.phi_hat = np.zeros((n, n))
        for (kx, ky), ph in zip(model.kf, model.phi):
            for sg in (1, -1):
                self.phi_hat[(sg * ky) % n, (sg * kx) % n] += ph / 2
        self.gain1 = self.gain(np.ones((n, n)))

    def _conv(self, FA, B):
        return np.real(np.fft.ifft2(FA * np.fft.fft2(B)))

    def gain(self, C: np.ndarray) -> np.ndarray:
        """sum_q W(q) [e(q).(s(k) + s(k-q))]^2 C(k-q)."""
        s = {"x": self.sx, "y": self.sy}
        out = np.zeros_like(C)
        for (i, j), mult in ((("x", "x"), 1.0), (("x", "y"), 2.0), (("y", "y"), 1.0)):
            FW, si, sj = self.FW[i + j], s[i], s[j]
            out += mult * (si * sj * self._conv(FW, C) + si * self._conv(FW, sj * C)
                           + sj * self._conv(FW, si * C) + self._conv(FW, si * sj * C))
        return out

    def rhs(self, C: np.ndarray) -> np.ndarray:
        return (self.gain(C) - self.gain1 * C - 2 * self.kappa * self.lam * C
                + self.phi_hat)


def lattice_spectrum(model: Kraichnan, kappa: float, tol: float = 1e-11,
                     maxiter: int = 20000) -> Tuple[np.ndarray, dict]:
    """Stationary C(k) of the closed lattice equation, by preconditioned GMRES.

    The mean (k = 0) is conserved and receives nothing, since e(q) is normal to
    s(q); it is pinned to zero.
    """
    from scipy.sparse.linalg import LinearOperator, gmres

    op = LatticeOperator(model, kappa)
    n = op.n
    diag = op.gain1 + 2 * kappa * op.lam
    diag[0, 0] = 1.0

    def mv(v):
        C = v.reshape(n, n)
        out = diag * C - op.gain(C)
        out[0, 0] = C[0, 0]
        return out.ravel()

    A = LinearOperator((n * n, n * n), matvec=mv, dtype=float)
    P = LinearOperator((n * n, n * n), matvec=lambda v: v / diag.ravel(), dtype=float)
    b = op.phi_hat.copy()
    b[0, 0] = 0.0
    it = [0]

    def cb(_):
        it[0] += 1

    x, info = gmres(A, b.ravel(), M=P, rtol=tol, restart=200, maxiter=maxiter,
                    callback=cb, callback_type="pr_norm")
    C = x.reshape(n, n)
    res = np.linalg.norm(op.rhs(C).ravel()[1:]) / np.linalg.norm(b)
    return C, dict(info=int(info), iterations=it[0], residual=float(res))


def shell_average(C2d: np.ndarray, kmax: int = None):
    """Sum of C(k) over integer shells |k| in [m - 1/2, m + 1/2): E(m), with sum E = variance."""
    n = C2d.shape[0]
    k = np.fft.fftfreq(n, d=1.0 / n)
    KY, KX = np.meshgrid(k, k, indexing="ij")
    m = np.rint(np.hypot(KX, KY)).astype(int)
    kmax = kmax or n // 2
    E = np.bincount(m.ravel(), weights=C2d.ravel(), minlength=kmax + 1)[:kmax + 1]
    return np.arange(kmax + 1), E
