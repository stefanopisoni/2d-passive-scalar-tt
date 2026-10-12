"""Explicit time stepping of a hybrid TT field, with dense references.

The design point is that *every* truncation is explicit and logged.  Compression
error is the whole question for TT time evolution, so nothing here compresses
behind the caller's back, and the diagnostics deliberately expose the variance
that truncation removes rather than hiding it (see :class:`Truncation`).

Two references are provided so the two error sources can be separated:

``dense_step``          the identical finite-difference scheme, run densely.
                        Comparing against it isolates *TT truncation error*.
``spectral_reference``  the analytic solution of the continuous equation.
                        Comparing against it gives the *total* error, including
                        the finite-difference spatial error.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Callable, Dict, List

import numpy as np
import torch

import config as cfg

sys.path.append(cfg.TENSNET_SRC)

import hybrid_operators as hop  # noqa: E402
import hybrid_ops as ho  # noqa: E402


# --------------------------------------------------------------------------
# truncation policy
# --------------------------------------------------------------------------
@dataclass
class Truncation:
    """Compression policy applied after each stage, plus a record of its cost.

    ``renormalize`` exists only for experiments; leaving it False is what makes
    the variance lost to truncation visible in the diagnostics instead of being
    silently restored.
    """

    bd_first: int = None
    bd_remaining: int = None
    eps: float = None
    renormalize: bool = False
    track_loss: bool = False      # measuring the loss costs two norms per call
    lost_norm2: float = 0.0       # cumulative squared l2 removed, ALL calls
    n_calls: int = 0

    # WARNING: lost_norm2 aggregates every compression, and variable_rhs performs
    # ~13 of them per RHS evaluation on *intermediates* (u * grad theta and
    # friends) whose norms are those of the right-hand side, not of theta.  It is
    # therefore NOT the numerical dissipation of the state and must not be
    # compared with kappa <|grad theta|^2>: doing so overstates it by orders of
    # magnitude (measured: 16.7x versus a true 0.002x).  Use
    # numerical_dissipation() below, which reads the actual variance budget.

    def __call__(self, h):
        # The two hybrid_norm calls needed to record the truncation loss cost
        # ~17% of total runtime, so they are opt-in: enable track_loss when the
        # dissipation budget is actually being measured, not for production runs.
        need = self.track_loss or self.renormalize
        before = ho.hybrid_norm(h) if need else 0.0
        out = ho.hybrid_compress(h, bd_first=self.bd_first,
                                 bd_remaining=self.bd_remaining, eps=self.eps)
        self.n_calls += 1
        if need:
            after = ho.hybrid_norm(out)
            self.lost_norm2 += max(before**2 - after**2, 0.0)
            if self.renormalize and after > 0:
                out = ho.hybrid_scale(out, before / after)
        return out

    def reset(self):
        self.lost_norm2 = 0.0
        self.n_calls = 0


def no_truncation(h):
    """Identity policy, for exactness checks."""
    return h


# --------------------------------------------------------------------------
# steppers
# --------------------------------------------------------------------------
def step_euler(h, rhs: Callable, dt: float, trunc: Callable):
    return trunc(ho.hybrid_axpy(1.0, h, dt, rhs(h)))


def step_rk2(h, rhs: Callable, dt: float, trunc: Callable):
    """Explicit midpoint."""
    k1 = rhs(h)
    mid = trunc(ho.hybrid_axpy(1.0, h, 0.5 * dt, k1))
    return trunc(ho.hybrid_axpy(1.0, h, dt, rhs(mid)))


def step_ssprk3(h, rhs: Callable, dt: float, trunc: Callable):
    """Strong-stability-preserving RK3 (Shu-Osher); the usual choice for advection."""
    u1 = trunc(ho.hybrid_axpy(1.0, h, dt, rhs(h)))
    u2 = trunc(ho.hybrid_axpy(0.75, h, 0.25, ho.hybrid_axpy(1.0, u1, dt, rhs(u1))))
    return trunc(ho.hybrid_axpy(1.0 / 3.0, h, 2.0 / 3.0,
                                ho.hybrid_axpy(1.0, u2, dt, rhs(u2))))


def step_rk4(h, rhs: Callable, dt: float, trunc: Callable):
    k1 = rhs(h)
    k2 = rhs(trunc(ho.hybrid_axpy(1.0, h, 0.5 * dt, k1)))
    k3 = rhs(trunc(ho.hybrid_axpy(1.0, h, 0.5 * dt, k2)))
    k4 = rhs(trunc(ho.hybrid_axpy(1.0, h, dt, k3)))
    acc = ho.hybrid_axpy(1.0, k1, 2.0, k2)
    acc = ho.hybrid_axpy(1.0, acc, 2.0, k3)
    acc = ho.hybrid_axpy(1.0, acc, 1.0, k4)
    return trunc(ho.hybrid_axpy(1.0, h, dt / 6.0, acc))


STEPPERS = {"euler": step_euler, "rk2": step_rk2, "ssprk3": step_ssprk3, "rk4": step_rk4}


def linear_rhs(op, N: int, trunc: Callable = no_truncation,
               fused: bool = False, head_op=None, sparse: bool = False) -> Callable:
    """RHS callable for a constant-coefficient operator held as one MPO.

    ``fused=True`` routes the head through an explicit 2^N x 2^N operator built
    once by :func:`hybrid_ops.fuse_head_operator`, rather than un-fusing the head
    into N binary legs on every application.
    """
    if fused:
        if head_op is None:
            head_op = ho.fuse_head_operator(op, N, sparse=sparse)

        def rhs(h):
            return trunc(ho.hybrid_apply_fused(op, h, N, head_op=head_op))
    else:
        def rhs(h):
            return trunc(ho.hybrid_apply(op, h, N))
    return rhs


# --------------------------------------------------------------------------
# dense reference, same discretisation
# --------------------------------------------------------------------------
def _apply_stencil_dense(f: np.ndarray, coef: Dict[int, float], axis: int) -> np.ndarray:
    out = np.zeros_like(f)
    for k, v in coef.items():
        if v:
            out += v * np.roll(f, -k, axis=axis)
    return out


def dense_rhs_adv_diff(cx: float, cy: float, kappa: float, dx: float,
                       adv_coef=None, diff_coef=None) -> Callable:
    """The same right-hand side as :func:`hybrid_operators.advection_diffusion_mpo`."""
    adv_coef = adv_coef or hop.D1_CENTRAL_2
    diff_coef = diff_coef or hop.D2_CENTRAL_2

    def rhs(f):
        out = np.zeros_like(f)
        if cx:
            out -= cx * _apply_stencil_dense(f, adv_coef, hop.AXIS_X) / dx
        if cy:
            out -= cy * _apply_stencil_dense(f, adv_coef, hop.AXIS_Y) / dx
        if kappa:
            out += kappa * (_apply_stencil_dense(f, diff_coef, hop.AXIS_X)
                            + _apply_stencil_dense(f, diff_coef, hop.AXIS_Y)) / dx**2
        return out
    return rhs


def dense_step(f: np.ndarray, rhs: Callable, dt: float, scheme: str = "rk2") -> np.ndarray:
    """One dense step with the same Butcher tableau as the hybrid stepper."""
    if scheme == "euler":
        return f + dt * rhs(f)
    if scheme == "rk2":
        return f + dt * rhs(f + 0.5 * dt * rhs(f))
    if scheme == "ssprk3":
        u1 = f + dt * rhs(f)
        u2 = 0.75 * f + 0.25 * (u1 + dt * rhs(u1))
        return f / 3.0 + (2.0 / 3.0) * (u2 + dt * rhs(u2))
    if scheme == "rk4":
        k1 = rhs(f)
        k2 = rhs(f + 0.5 * dt * k1)
        k3 = rhs(f + 0.5 * dt * k2)
        k4 = rhs(f + dt * k3)
        return f + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
    raise ValueError(f"unknown scheme {scheme}")


def spectral_reference(f0: np.ndarray, t: float, cx: float, cy: float,
                       kappa: float, dx: float) -> np.ndarray:
    """Exact solution of the *continuous* advection-diffusion problem at time t.

    Periodic domain of side ``n*dx``; carries no spatial or temporal
    discretisation error, so it measures the total error of the scheme.
    """
    n = f0.shape[0]
    kk = 2.0 * np.pi * np.fft.fftfreq(n, d=dx)
    KY, KX = np.meshgrid(kk, kk, indexing="ij")   # axis 0 = rows = y
    sym = np.exp(-1j * (KX * cx + KY * cy) * t - kappa * (KX**2 + KY**2) * t)
    return np.real(np.fft.ifft2(np.fft.fft2(f0) * sym))


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def seam_jump_ratio(f: np.ndarray, L: int, N: int) -> float:
    """Mean |first difference| across seam lines, relative to off-seam.

    1.0 means the coarse-block boundaries are indistinguishable from the rest of
    the field; larger values quantify the compression-induced discontinuity.
    Averaged over both axes, since the seam structure is isotropic.
    """
    idx = ho.seam_indices(L, N).numpy()
    dx_ = np.abs(f - np.roll(f, -1, axis=hop.AXIS_X))
    dy_ = np.abs(f - np.roll(f, -1, axis=hop.AXIS_Y))
    on = 0.5 * (dx_[:, idx].mean() + dy_[idx, :].mean())
    off = 0.5 * (np.delete(dx_, idx, axis=1).mean() + np.delete(dy_, idx, axis=0).mean())
    return float(on / off) if off > 0 else float("nan")


@dataclass
class History:
    t: List[float] = field(default_factory=list)
    mass: List[float] = field(default_factory=list)
    variance: List[float] = field(default_factory=list)
    max_bond: List[int] = field(default_factory=list)
    seam: List[float] = field(default_factory=list)
    err_vs_dense: List[float] = field(default_factory=list)
    err_vs_exact: List[float] = field(default_factory=list)

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {k: np.asarray(v) for k, v in self.__dict__.items()}


def record(hist: History, t: float, h, L: int, N: int,
           dense: np.ndarray = None, exact: np.ndarray = None) -> np.ndarray:
    """Append one row of diagnostics; returns the densified field."""
    f = ho.hybrid_to_field(h, L).numpy()
    hist.t.append(t)
    hist.mass.append(float(f.mean()))
    hist.variance.append(float(f.var()))
    hist.max_bond.append(int(max(ho.bond_dims(h))))
    hist.seam.append(seam_jump_ratio(f, L, N))
    denom = np.abs(dense).max() if dense is not None else 1.0
    hist.err_vs_dense.append(float(np.abs(f - dense).max() / denom) if dense is not None else np.nan)
    if exact is not None:
        hist.err_vs_exact.append(float(np.abs(f - exact).max() / np.abs(exact).max()))
    else:
        hist.err_vs_exact.append(np.nan)
    return f


# --------------------------------------------------------------------------
# driver
# --------------------------------------------------------------------------
def stable_dt(cx: float, cy: float, kappa: float, dx: float,
              cfl: float = 0.2, diff_number: float = 0.1) -> float:
    """Largest dt satisfying both the advective and the 2-D diffusive limit.

    Explicit diffusion in 2D needs ``dt <= dx^2 / (4 kappa)``; taking dt from the
    CFL condition alone silently produces an unstable scheme whenever the grid
    is fine enough, which is easy to do since dx^2 shrinks quadratically.
    """
    limits = []
    speed = max(abs(cx), abs(cy))
    if speed:
        limits.append(cfl * dx / speed)
    if kappa:
        limits.append(diff_number * dx**2 / kappa)
    if not limits:
        raise ValueError("no advection and no diffusion")
    return min(limits)


def evolve(h0, L: int, N: int, rhs_mpo, dt: float, n_steps: int,
           trunc: Truncation, scheme: str = "rk2", seam_mpo=None,
           dense_f0: np.ndarray = None, dense_rhs: Callable = None,
           exact_fn: Callable = None, every: int = 1,
           conserve_mass: bool = False, seam_every: int = 1,
           fused: bool = False, sparse: bool = False):
    """Integrate a hybrid field, recording diagnostics every ``every`` steps.

    ``seam_mpo`` applies the isotropic seam smoother every ``seam_every`` steps.
    Note that the smoother is *not* idempotent, so applying it in-loop acts as a
    per-step low-pass filter; measurements show that at ``seam_every=1`` it
    removes far more variance than it repairs.  Prefer applying it to the output
    for analysis rather than to the evolving state -- the seam is an artifact of
    the representation, and feeding its repair back into the dynamics compounds.

    ``conserve_mass`` restores the initial mean after every step (rank-1, and
    exact for advection-diffusion, which conserves total mass).
    """
    step = STEPPERS[scheme]
    rhs = linear_rhs(rhs_mpo, N, trunc=trunc, fused=fused, sparse=sparse)
    seam_head = (ho.fuse_head_operator(seam_mpo, N, sparse=sparse)
                 if (fused and seam_mpo is not None) else None)
    h = h0
    mean0 = ho.hybrid_sum(h0) / 4**L
    dense = None if dense_f0 is None else dense_f0.copy()
    hist = History()
    record(hist, 0.0, h, L, N, dense, None if exact_fn is None else exact_fn(0.0))

    for i in range(1, n_steps + 1):
        h = step(h, rhs, dt, trunc)
        if seam_mpo is not None and i % seam_every == 0:
            h = trunc(ho.hybrid_apply_fused(seam_mpo, h, N, head_op=seam_head)
                      if fused else ho.hybrid_apply(seam_mpo, h, N))
        if conserve_mass:
            # must come *after* the truncation, which would otherwise reintroduce
            # the mass error the projection just removed. Costs one bond, which
            # the next step's truncation reclaims.
            h = ho.project_mass(h, L, N, mean0)
        if dense is not None and dense_rhs is not None:
            dense = dense_step(dense, dense_rhs, dt, scheme)
        if i % every == 0 or i == n_steps:
            record(hist, i * dt, h, L, N, dense,
                   None if exact_fn is None else exact_fn(i * dt))
    return h, hist, dense


# --------------------------------------------------------------------------
# variable-coefficient transport
# --------------------------------------------------------------------------
def velocity_from_streamfunction(psi, L: int, N: int, dx: float,
                                 coef: Dict[int, float] = None,
                                 trunc: Callable = no_truncation):
    """Build (ux, uy) = (d(psi)/dy, -d(psi)/dx) with the *discrete* operators.

    Deriving the velocity from a streamfunction with the same stencils used by
    the solver makes it divergence-free to round-off at the discrete level,

        Dx ux + Dy uy = Dx Dy psi - Dy Dx psi = 0,

    because Dx and Dy act on different axes and commute exactly.  Sampling
    sin/cos analytically instead would leave a truncation-order divergence, and
    the skew-symmetry that conserves scalar variance would then only hold
    approximately.
    """
    coef = coef or hop.D1_CENTRAL_2
    Dx = hop.derivative_mpo(L, hop.AXIS_X, coef, dx=dx)
    Dy = hop.derivative_mpo(L, hop.AXIS_Y, coef, dx=dx)
    ux = trunc(ho.hybrid_apply(Dy, psi, N))
    uy = trunc(ho.hybrid_scale(ho.hybrid_apply(Dx, psi, N), -1.0))
    return ux, uy


def variable_rhs(ux, uy, L: int, N: int, dx: float, kappa: float = 0.0,
                 form: str = "skew", adv_coef: Dict[int, float] = None,
                 diff_coef: Dict[int, float] = None, forcing=None,
                 compress_inner: bool = True,
                 trunc: Callable = no_truncation) -> Callable:
    """RHS of the passive scalar with a prescribed, space-dependent velocity.

    ``form`` selects the discretisation of the transport term:

    ``"advective"``     -u . grad(theta)
    ``"conservative"``  -div(u theta)
    ``"skew"``          the average of the two (default)

    All three coincide in the continuum for incompressible u but not after
    discretisation.  The skew form is the one that inherits the exact
    antisymmetry of the continuous operator and therefore conserves
    <theta^2> best; it is the standard choice for this reason.

    Every intermediate is compressed: a Hadamard product multiplies bond
    dimensions (chi_u * chi_theta), so leaving the stages uncompressed makes the
    final sweep cost O(chi^3) on a bond dimension several hundred wide.
    """
    adv_coef = adv_coef or hop.D1_CENTRAL_2
    diff_coef = diff_coef or hop.D2_CENTRAL_2
    Dx = hop.derivative_mpo(L, hop.AXIS_X, adv_coef, dx=dx)
    Dy = hop.derivative_mpo(L, hop.AXIS_Y, adv_coef, dx=dx)
    lap = hop.laplacian_mpo(L, diff_coef, dx=dx) if kappa else None
    if form not in ("advective", "conservative", "skew"):
        raise ValueError(f"unknown form {form!r}")

    # Compress after EVERY factor, never letting two rank-multiplying operations
    # stack.  Applying D then multiplying by u without an intermediate compress
    # peaks at r_D * chi_u * chi (12 chi here) instead of max(r_D, chi_u) * chi
    # (4 chi); since the sweep costs O(chi^3) that is a ~27x difference.
    # How often to compress inside the transport term. Compressing after every
    # factor keeps the peak bond dimension at max(r_D, chi_u)*chi instead of
    # r_D*chi_u*chi, but each call is a full two-sweep and the sweeps turn out to
    # be dominated by per-call overhead rather than flops, so fewer-and-larger
    # can win. Measured, not assumed; see the granularity comparison in the notes.
    inner = trunc if compress_inner else (lambda z: z)

    def _adv_x(h):
        return inner(ho.hybrid_hadamard(ux, inner(ho.hybrid_apply(Dx, h, N))))

    def _adv_y(h):
        return inner(ho.hybrid_hadamard(uy, inner(ho.hybrid_apply(Dy, h, N))))

    def _con_x(h):
        return inner(ho.hybrid_apply(Dx, inner(ho.hybrid_hadamard(ux, h)), N))

    def _con_y(h):
        return inner(ho.hybrid_apply(Dy, inner(ho.hybrid_hadamard(uy, h)), N))

    def rhs(h):
        # Accumulate every contribution exactly and truncate once at the end.
        # Truncating after each partial sum instead costs three extra full
        # two-sweep compressions per RHS evaluation for no accuracy gain, since
        # the sums themselves are exact.
        terms = []
        if form in ("advective", "skew"):
            w = 1.0 if form == "advective" else 0.5
            terms += [(-w, _adv_x(h)), (-w, _adv_y(h))]
        if form in ("conservative", "skew"):
            w = 1.0 if form == "conservative" else 0.5
            terms += [(-w, _con_x(h)), (-w, _con_y(h))]
        if kappa:
            terms.append((kappa, ho.hybrid_apply(lap, h, N)))
        if forcing is not None:
            # without a source a passive scalar simply decays to nothing, so a
            # sustained cascade -- the regime the statistics describe -- needs
            # steady large-scale injection.
            terms.append((1.0, forcing))
        acc = ho.hybrid_scale(terms[0][1], terms[0][0])
        for c, t in terms[1:]:
            acc = ho.hybrid_add(acc, ho.hybrid_scale(t, c))
        return trunc(acc)

    return rhs


def dense_variable_rhs(ux: np.ndarray, uy: np.ndarray, dx: float,
                       kappa: float = 0.0, form: str = "skew",
                       adv_coef=None, diff_coef=None, forcing=None) -> Callable:
    """Dense counterpart of :func:`variable_rhs`, same stencils and same form."""
    adv_coef = adv_coef or hop.D1_CENTRAL_2
    diff_coef = diff_coef or hop.D2_CENTRAL_2

    def advective(f):
        return -(ux * _apply_stencil_dense(f, adv_coef, hop.AXIS_X)
                 + uy * _apply_stencil_dense(f, adv_coef, hop.AXIS_Y)) / dx

    def conservative(f):
        return -(_apply_stencil_dense(ux * f, adv_coef, hop.AXIS_X)
                 + _apply_stencil_dense(uy * f, adv_coef, hop.AXIS_Y)) / dx

    def rhs(f):
        if form == "advective":
            t = advective(f)
        elif form == "conservative":
            t = conservative(f)
        else:
            t = 0.5 * (advective(f) + conservative(f))
        if kappa:
            t = t + kappa * (_apply_stencil_dense(f, diff_coef, hop.AXIS_X)
                             + _apply_stencil_dense(f, diff_coef, hop.AXIS_Y)) / dx**2
        if forcing is not None:
            t = t + forcing
        return t
    return rhs


def numerical_dissipation(t, var_run, var_ref):
    """Numerical dissipation from the variance budget: the *extra* variance decay.

    ``var_run`` and ``var_ref`` are variance histories on the same times ``t``,
    for the truncated run and for an essentially converged reference.  Returns
    ``(d/dt <theta^2>_ref - d/dt <theta^2>_run)/2``, directly comparable with
    :func:`scalar_dissipation`.

    This is the trustworthy measurement.  Summing truncation losses instead
    conflates the state with the right-hand-side intermediates.
    """
    t = np.asarray(t, dtype=float)
    return (np.gradient(np.asarray(var_run), t)
            - np.gradient(np.asarray(var_ref), t)) * -0.5


def scalar_dissipation(f: np.ndarray, kappa: float, dx: float,
                       coef: Dict[int, float] = None) -> float:
    """Physical dissipation rate kappa * <|grad theta|^2> from the variance budget."""
    coef = coef or hop.D1_CENTRAL_2
    gx = _apply_stencil_dense(f, coef, hop.AXIS_X) / dx
    gy = _apply_stencil_dense(f, coef, hop.AXIS_Y) / dx
    return float(kappa * np.mean(gx**2 + gy**2))
