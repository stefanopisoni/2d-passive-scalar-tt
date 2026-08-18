"""MPO builders for the hybrid TT solver, in the interleaved (Morton) layout.

Axis convention, fixed by :func:`hybrid_ops.shuffle_axis`
---------------------------------------------------------
The field is stored as ``field[row, col]``.  After interleaving::

    leg:  0     1     2     3     ...
          row_0 col_0 row_1 col_1 ...        (level 0 = MSB = coarsest)

so **even legs carry the row index and odd legs the column index**.  Following
``generate_data.py``, *x* means the column direction (axis 1) and *y* the row
direction (axis 0).  Both are exported as :data:`AXIS_X` / :data:`AXIS_Y` so no
caller has to remember this.

Every builder here passes ``arch="sequential"`` explicitly.  tensnet's global
default is ``"stacked"`` (the non-interleaved layout), and picking that up by
accident would not raise -- it would silently produce the wrong physics.
Likewise, stencil coefficients are always passed explicitly rather than letting
``get_diff_mpo`` read the global ``low_acc`` flag.
"""
from __future__ import annotations

import sys
from typing import Dict, List, Sequence

import torch

import config as cfg

sys.path.append(cfg.TENSNET_SRC)
import tensnet as tt  # noqa: E402
from tensnet.linalg.diff_operators import construct_toeplitz, get_2D_operator  # noqa: E402

import hybrid_ops as ho  # noqa: E402

AXIS_Y = 0  # rows, even interleaved legs
AXIS_X = 1  # columns, odd interleaved legs


# --------------------------------------------------------------------------
# stencils
# --------------------------------------------------------------------------
def stencil(entries: Dict[int, float]) -> Dict[int, float]:
    """Pad a stencil to the -4..4 key range ``construct_toeplitz`` indexes."""
    c = {i: 0.0 for i in range(-4, 5)}
    for k, v in entries.items():
        if not -4 <= k <= 4:
            raise ValueError(f"offset {k} outside the supported range -4..4")
        c[k] = float(v)
    return c


#: centred first derivative, 2nd order:  (f(x+1) - f(x-1)) / 2
D1_CENTRAL_2 = stencil({1: 0.5, -1: -0.5})
#: centred first derivative, 4th order
D1_CENTRAL_4 = stencil({1: 2 / 3, -1: -2 / 3, 2: -1 / 12, -2: 1 / 12})
#: centred second derivative, 2nd order:  f(x+1) - 2 f(x) + f(x-1)
D2_CENTRAL_2 = stencil({0: -2.0, 1: 1.0, -1: 1.0})
#: centred second derivative, 4th order
D2_CENTRAL_4 = stencil({0: -5 / 2, 1: 4 / 3, -1: 4 / 3, 2: -1 / 12, -2: -1 / 12})
#: upwind first derivative for positive advection speed:  f(x) - f(x-1)
D1_UPWIND_POS = stencil({0: 1.0, -1: -1.0})
#: upwind first derivative for negative advection speed:  f(x+1) - f(x)
D1_UPWIND_NEG = stencil({1: 1.0, 0: -1.0})
#: two-neighbour average, used by the seam smoother
NEIGHBOUR_AVG = stencil({1: 0.5, -1: 0.5})


def translation(cells: int) -> Dict[int, float]:
    """Stencil translating the field by ``+cells`` grid points.

    ``(T f)[i] = f[i - cells]``, matching ``numpy.roll(f, cells)``, so a
    positive value moves structures toward increasing index.
    """
    if not -4 <= cells <= 4:
        raise ValueError("compose several translations for shifts beyond 4 cells")
    return stencil({-cells: 1.0})


# --------------------------------------------------------------------------
# generic MPO algebra (exact -- tensnet's MPO.__add__/__mul__ route through
# MPS.__add__/__mul__, which compress silently at COMP_ACC)
# --------------------------------------------------------------------------
def mpo_tensors(mpo) -> List[torch.Tensor]:
    """Plain list of 4-leg cores ``(left, in, right, out)``."""
    return [node.tensor for node in mpo]


def mpo_from_tensors(W: Sequence[torch.Tensor]):
    return tt.list_to_mpo(list(W))


def mpo_add(a, b):
    """Exact sum of two MPOs. Bond dimensions add."""
    A, B = mpo_tensors(a), mpo_tensors(b)
    if len(A) != len(B):
        raise ValueError("operands must have the same number of sites")
    n = len(A)
    out = []
    for k, (x, y) in enumerate(zip(A, B)):
        if x.shape[1] != y.shape[1] or x.shape[3] != y.shape[3]:
            raise ValueError(f"physical dim mismatch at site {k}")
        if k == 0:
            out.append(torch.cat([x, y], dim=2))
        elif k == n - 1:
            out.append(torch.cat([x, y], dim=0))
        else:
            lx, i, rx, o = x.shape
            ly, _, ry, _ = y.shape
            top = torch.cat([x, x.new_zeros(lx, i, ry, o)], dim=2)
            bot = torch.cat([y.new_zeros(ly, i, rx, o), y], dim=2)
            out.append(torch.cat([top, bot], dim=0))
    return mpo_from_tensors(out)


def mpo_scale(a, c: float):
    """Exact scalar multiple."""
    W = [T.clone() for T in mpo_tensors(a)]
    W[-1] = W[-1] * c
    return mpo_from_tensors(W)


def mpo_compose(a, b):
    """Exact operator product: ``mpo_compose(A, B) f == A (B f)``. Bonds multiply."""
    A, B = mpo_tensors(a), mpo_tensors(b)
    if len(A) != len(B):
        raise ValueError("operands must have the same number of sites")
    out = []
    for k, (x, y) in enumerate(zip(A, B)):
        # x = (p, k_in, q, o) ; y = (s, c, t, k_out) ; contract x's input with y's output
        if x.shape[1] != y.shape[3]:
            raise ValueError(f"contraction dim mismatch at site {k}")
        t = torch.einsum("pkqo,sctk->pscqto", x, y)
        p, s, c, q, tt_, o = t.shape
        out.append(t.reshape(p * s, c, q * tt_, o))
    return mpo_from_tensors(out)


def mpo_compress(a, bdmax: int = None, eps: float = None):
    """Two-sweep compression of an MPO, reusing the MPS core sweep."""
    W = mpo_tensors(a)
    shapes = [(T.shape[1], T.shape[3]) for T in W]
    cores = [T.permute(0, 1, 3, 2).reshape(T.shape[0], T.shape[1] * T.shape[3], T.shape[2])
             for T in W]
    cores = ho.compress_cores(cores, caps=[bdmax] * (len(cores) - 1), eps=eps)
    out = []
    for T, (i, o) in zip(cores, shapes):
        l, _, r = T.shape
        out.append(T.reshape(l, i, o, r).permute(0, 1, 3, 2))
    return mpo_from_tensors(out)


def mpo_to_dense(a) -> torch.Tensor:
    """Dense matrix of an MPO. ``MPO.contract`` assumes a uniform physical dim."""
    W = mpo_tensors(a)
    out = W[0]
    for T in W[1:]:
        l, i, r, o = out.shape
        _, j, s, p = T.shape
        out = torch.einsum("liro,rjsp->lijsop", out, T).reshape(l, i * j, s, o * p)
    l, i, r, o = out.shape
    return out.reshape(i, o).T.contiguous()


# --------------------------------------------------------------------------
# 1-D building blocks
# --------------------------------------------------------------------------
def toeplitz_1d(L: int, coef: Dict[int, float], bc: str = "periodic"):
    """1-D Toeplitz (circulant, for ``bc='periodic'``) operator on 2^L points."""
    return construct_toeplitz(L, coef, bc)


def diagonal_1d(mats: Sequence[torch.Tensor]):
    """Bond-dimension-1 MPO from one 2x2 matrix per bit, MSB first.

    Core layout is ``(left, in, right, out)`` with axis 1 the input (column) leg
    and axis 3 the output (row) leg, so a matrix ``m`` is stored transposed.
    """
    W = []
    for m in mats:
        if tuple(m.shape) != (2, 2):
            raise ValueError("each factor must be 2x2")
        W.append(m.T.reshape(1, 2, 1, 2).clone())
    return mpo_from_tensors(W)


def promote_2d(P, axis: int, L: int):
    """Lift a 1-D MPO to the interleaved 2-D layout along ``axis``."""
    if axis not in (AXIS_Y, AXIS_X):
        raise ValueError(f"axis must be {AXIS_Y} (rows/y) or {AXIS_X} (cols/x)")
    return get_2D_operator(P, axis, [L, L], arch="sequential")


def identity_2d(L: int):
    """Identity on the full 2L-leg interleaved layout."""
    return tt.identity_mpo(2 * L, dtype=torch.get_default_dtype())


# --------------------------------------------------------------------------
# differential operators
# --------------------------------------------------------------------------
def derivative_mpo(L: int, axis: int, coef: Dict[int, float] = D1_CENTRAL_2,
                   dx: float = 1.0, bc: str = "periodic", compress_eps: float = 1e-14):
    """Finite-difference derivative along ``axis``, divided by ``dx``.

    ``construct_toeplitz`` always returns bond dimension 9 regardless of how
    many stencil terms are non-zero, so the result is compressed once here.
    """
    P = toeplitz_1d(L, coef, bc)
    op = promote_2d(P, axis, L)
    op = mpo_compress(op, eps=compress_eps)
    return mpo_scale(op, 1.0 / dx) if dx != 1.0 else op


def laplacian_mpo(L: int, coef: Dict[int, float] = D2_CENTRAL_2, dx: float = 1.0,
                  bc: str = "periodic", compress_eps: float = 1e-14):
    """``d2/dx2 + d2/dy2`` as a single compressed MPO."""
    dxx = promote_2d(toeplitz_1d(L, coef, bc), AXIS_X, L)
    dyy = promote_2d(toeplitz_1d(L, coef, bc), AXIS_Y, L)
    op = mpo_compress(mpo_add(dxx, dyy), eps=compress_eps)
    return mpo_scale(op, 1.0 / dx**2) if dx != 1.0 else op


def translation_mpo(L: int, axis: int, cells: int = 1, compress_eps: float = 1e-14):
    """Exact periodic translation by ``cells`` grid points along ``axis``.

    A permutation matrix: bond dimension 2 after compression, and applying it
    leaves the TT ranks of the state unchanged.  This makes constant-velocity
    advection at CFL = 1 exactly representable, so any drift observed in such a
    run is attributable entirely to truncation.
    """
    op = promote_2d(toeplitz_1d(L, translation(cells), "periodic"), axis, L)
    return mpo_compress(op, eps=compress_eps)


def advection_diffusion_mpo(L: int, cx: float = 0.0, cy: float = 0.0,
                            kappa: float = 0.0, dx: float = 1.0,
                            adv_coef: Dict[int, float] = D1_CENTRAL_2,
                            diff_coef: Dict[int, float] = D2_CENTRAL_2,
                            compress_eps: float = 1e-14):
    """RHS operator of ``d(theta)/dt = -cx d/dx - cy d/dy + kappa laplacian``.

    Constant coefficients, so the whole right-hand side collapses into a single
    MPO and one step costs one application.
    """
    terms = []
    if cx:
        terms.append(mpo_scale(derivative_mpo(L, AXIS_X, adv_coef, dx=dx,
                                              compress_eps=compress_eps), -cx))
    if cy:
        terms.append(mpo_scale(derivative_mpo(L, AXIS_Y, adv_coef, dx=dx,
                                              compress_eps=compress_eps), -cy))
    if kappa:
        terms.append(mpo_scale(laplacian_mpo(L, diff_coef, dx=dx,
                                             compress_eps=compress_eps), kappa))
    if not terms:
        raise ValueError("all of cx, cy, kappa are zero")
    op = terms[0]
    for t in terms[1:]:
        op = mpo_add(op, t)
    return mpo_compress(op, eps=compress_eps)


# --------------------------------------------------------------------------
# isotropic seam smoother
# --------------------------------------------------------------------------
def seam_projector(L: int, N: int, axis: int):
    """Projector onto the coarse-block boundary lines of ``axis``.

    A line index sits on a seam iff its ``L - N/2`` fine bits are all 1, i.e.
    iff ``(i + 1) % 2^(L - N/2) == 0`` -- the last line of each coarse block.
    Bond dimension 1: it is a product of single-bit factors.
    """
    ho._check_LN(L, N)
    n_coarse = N // 2
    eye = torch.eye(2, dtype=torch.get_default_dtype())
    one = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.get_default_dtype())
    mats = [eye if j < n_coarse else one for j in range(L)]
    return promote_2d(diagonal_1d(mats), axis, L)


def seam_smoother_mpo(L: int, N: int, compress_eps: float = 1e-13):
    """Isotropic seam repair as a single MPO.

    Reproduces the ``_int64_xy`` treatment in ``generate_data.py``: every cell on
    a flagged **column** is replaced by the mean of its east/west neighbours,
    every cell on a flagged **row** by the mean of its north/south neighbours,
    and cells on both (the block corners) by the mean of those two
    interpolants.  All neighbour values are taken from the *input* field, and
    wrapping is periodic.

    With ``Px``/``Py`` the seam projectors, ``Qx = I - Px``, ``Qy = I - Py`` and
    ``Ax``/``Ay`` the periodic two-neighbour averages::

        S = Qx Qy  +  Px Qy Ax  +  Qx Py Ay  +  Px Py (Ax + Ay)/2

    Applied once per step this is a linear correction, not a projection: it is
    not idempotent, so repeated application keeps smoothing.
    """
    I = identity_2d(L)
    Px = seam_projector(L, N, AXIS_X)
    Py = seam_projector(L, N, AXIS_Y)
    Qx = mpo_add(I, mpo_scale(Px, -1.0))
    Qy = mpo_add(I, mpo_scale(Py, -1.0))
    Ax = promote_2d(toeplitz_1d(L, NEIGHBOUR_AVG, "periodic"), AXIS_X, L)
    Ay = promote_2d(toeplitz_1d(L, NEIGHBOUR_AVG, "periodic"), AXIS_Y, L)
    Ax = mpo_compress(Ax, eps=1e-14)
    Ay = mpo_compress(Ay, eps=1e-14)

    interior = mpo_compose(Qx, Qy)
    col_only = mpo_compose(mpo_compose(Px, Qy), Ax)
    row_only = mpo_compose(mpo_compose(Qx, Py), Ay)
    corner = mpo_compose(mpo_compose(Px, Py),
                         mpo_scale(mpo_add(Ax, Ay), 0.5))

    S = mpo_add(mpo_add(interior, col_only), mpo_add(row_only, corner))
    return mpo_compress(S, eps=compress_eps)


def seam_smooth_dense(field: torch.Tensor, L: int, N: int) -> torch.Tensor:
    """Reference implementation of the isotropic seam repair, on a dense field.

    Mirrors ``generate_data.py``'s ``_int64_xy`` block; used to validate
    :func:`seam_smoother_mpo`.
    """
    ho._check_LN(L, N)
    n = 2**L
    idx = ho.seam_indices(L, N).numpy()
    col_flag = torch.zeros(n, dtype=torch.bool)
    row_flag = torch.zeros(n, dtype=torch.bool)
    col_flag[idx] = True
    row_flag[idx] = True

    x_interp = 0.5 * (torch.roll(field, 1, dims=1) + torch.roll(field, -1, dims=1))
    y_interp = 0.5 * (torch.roll(field, 1, dims=0) + torch.roll(field, -1, dims=0))

    C = col_flag[None, :].expand(n, n)
    R = row_flag[:, None].expand(n, n)
    out = field.clone()
    out[C & ~R] = x_interp[C & ~R]
    out[R & ~C] = y_interp[R & ~C]
    out[C & R] = 0.5 * (x_interp + y_interp)[C & R]
    return out
