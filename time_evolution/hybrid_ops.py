"""Hybrid TT algebra: dense head + compressed tail.

Everything here is built *on top of* tensnet using its public API only; no
tensnet source is modified, and no tensnet global state (``ARCH_2D``,
``COMP_ACC``, ``BD_LIMIT``, ``low_acc``) is read or written.  Truncation is
always explicit and passed in by the caller.

Conventions (matching ``generate_data.py`` / ``figure5.py`` / the paper)
-----------------------------------------------------------------------
``L``   bits per axis, so the grid is 2^L x 2^L.
``2L``  total physical legs, in the interleaved (Morton) order produced by
        :func:`shuffle_axis`::

            leg:  0      1      2      3     ...
                  row_0  col_0  row_1  col_1 ...     (level 0 = MSB = coarsest)

``N``   legs carried by the dense head; must be even, so the head holds the
        N/2 most significant bits of *each* axis and therefore partitions the
        domain into 2^(N/2) x 2^(N/2) coarse blocks.

A hybrid MPS has ``1 + (2L - N)`` sites: site 0 has physical dimension 2^N,
every other site has physical dimension 2.  The coarse-block seam period, in
grid cells and identical for rows and columns, is ``2^(L - N/2)``.

Why several tensnet routines are deliberately not used
------------------------------------------------------
``MPS.__add__`` / ``__mul__`` / ``MPO.apply`` all call ``compress()`` with the
global ``COMP_ACC`` (1e-6, an *l2* tolerance).  Measured on a hybrid MPS a
single such call injects ~2e-4 max-norm error -- the wrong error norm for a
study about intermittency tails.  ``MPO.__matmul__`` additionally requires the
MPO and MPS to have equal site counts, which a hybrid MPS never does.
``MPO.contract()`` assumes a uniform physical dimension.  ``tt.svd``
renormalises the singular values after truncation by default, which would fake
conservation of variance under time stepping.  Hence the local reimplementations
below.
"""
from __future__ import annotations

import os
import sys

# This package lives one level below the repository root, which still holds the
# paper-side modules it depends on (config.py for TENSNET_SRC, hybrid_MPS.py for
# the legacy encoder).  Put both this directory and the root on sys.path so the
# modules import the same way whether they are run as scripts, imported from a
# test, or imported from the repository root.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_HERE, _ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from typing import Iterable, List, Sequence

import numpy as np
import scipy.linalg as sla
import torch

import config as cfg

sys.path.append(cfg.TENSNET_SRC)
import tensnet as tt  # noqa: E402


# --------------------------------------------------------------------------
# index layout
# --------------------------------------------------------------------------
def shuffle_axis(A: torch.Tensor, L: int) -> torch.Tensor:
    """Interleave the row/column bits of a 2^L x 2^L field (Morton order).

    Identical to the helper in ``generate_data.py``; duplicated here because
    importing that module would execute the whole data pipeline.
    """
    new_axis = []
    for x in range(L):
        new_axis += [x, x + L]
    return A.reshape([2] * 2 * L).permute(new_axis).reshape(2**L, 2**L)


def unshuffle_axis(A: torch.Tensor, L: int) -> torch.Tensor:
    """Inverse of :func:`shuffle_axis`."""
    axis_even_odd = [x for x in range(2 * L) if x % 2 == 0] + [
        x for x in range(2 * L) if x % 2 == 1
    ]
    return A.reshape([2] * (2 * L)).permute(axis_even_odd).reshape(2**L, 2**L)


def seam_period(L: int, N: int) -> int:
    """Coarse-block width in grid cells; same for rows and columns.

    NOTE the exponent is ``L - N/2``, not ``N/2``: the head holds the N/2 most
    significant bits of each axis, so the *fine* part of each axis has
    ``L - N/2`` bits.  (These coincide only at N = L.)
    """
    _check_LN(L, N)
    return 2 ** (L - N // 2)


def seam_indices(L: int, N: int) -> torch.Tensor:
    """Row/column indices sitting on a coarse-block boundary (last cell of each block).

    Empty when N=0: with no head there are no coarse blocks and so no seams.
    (The formula would otherwise return the periodic wrap boundary, which is
    not a seam.)
    """
    if N == 0:
        return torch.empty(0, dtype=torch.long)
    p = seam_period(L, N)
    idx = torch.arange(2**L)
    return idx[(idx + 1) % p == 0]


def _check_LN(L: int, N: int) -> None:
    """``N=0`` is permitted and means *no head*: a plain QTT.

    With N=0 the construction below emits a trivial (1,1,1) leading core and
    2L binary tail cores, which is exactly ``tt.vector_to_mps(vec, 2, 2L)`` --
    the standard MPS the paper compares against. Keeping it inside the same
    code path matters: a separate implementation would differ in compression
    policy and contraction order, and those differences would contaminate any
    hybrid-versus-standard comparison.
    """
    if N % 2:
        raise ValueError(f"N={N} must be even so the head holds N/2 bits of each axis")
    if not 0 <= N < 2 * L:
        raise ValueError(f"N={N} must satisfy 0 <= N < 2L = {2 * L}")


# --------------------------------------------------------------------------
# construction / inspection
# --------------------------------------------------------------------------
def tensors(mps) -> List[torch.Tensor]:
    """Plain list of the (left, phys, right) core tensors of an MPS."""
    return [node.tensor for node in mps]


def from_tensors(A: Sequence[torch.Tensor]):
    """Wrap a list of cores back into a tensnet MPS."""
    return tt.list_to_mps(list(A))


def validate_hybrid(mps, L: int, N: int) -> None:
    """Raise unless ``mps`` has the hybrid shape implied by (L, N)."""
    _check_LN(L, N)
    A = tensors(mps)
    n_expected = 1 + (2 * L - N)
    if len(A) != n_expected:
        raise ValueError(f"expected {n_expected} sites for L={L}, N={N}, got {len(A)}")
    if A[0].shape[1] != 2**N:
        raise ValueError(f"head physical dim {A[0].shape[1]} != 2^N = {2**N}")
    for k, T in enumerate(A[1:], start=1):
        if T.shape[1] != 2:
            raise ValueError(f"site {k} physical dim {T.shape[1]} != 2")
    if A[0].shape[0] != 1 or A[-1].shape[2] != 1:
        raise ValueError("open boundary bonds must be 1")
    for k in range(len(A) - 1):
        if A[k].shape[2] != A[k + 1].shape[0]:
            raise ValueError(f"bond mismatch between sites {k} and {k + 1}")


def vector_to_hybrid(vec: torch.Tensor, n_legs: int, N: int, bd_first: int,
                     bd_remaining: int = None, eps: float = None,
                     renormalize: bool = False):
    """Hybrid TT decomposition of an interleaved state vector.

    Same construction as :func:`hybrid_MPS.vector_to_hybrid_mps`, but going
    through :func:`truncated_svd`.  Two behaviours therefore differ from the
    legacy path, both deliberately:

    ``eps``
        ``tt.svd`` picks the rank from ``cumsum(S**2) < (1 - eps) * sum(S**2)``,
        i.e. ``eps`` is a *squared-energy* fraction, so the achievable amplitude
        error is only ``~sqrt(eps)`` and saturates near 1e-8 (below ~1e-16,
        ``1 - eps`` is no longer representable).  Here ``eps`` is an amplitude
        tolerance that can actually be tightened.

    ``renormalize``
        ``tt.svd`` defaults to ``True``, rescaling the kept singular values back
        up to the full norm at *every* split, so a rank-capped legacy hybrid TT
        matches the original field's l2 norm exactly rather than being the best
        rank-r approximation.  We default to ``False``.  Set ``True`` to
        reproduce ``vector_to_hybrid_mps`` bit-for-bit.
    """
    if bd_remaining is None:
        bd_remaining = bd_first
    n_tail = n_legs - N
    if n_tail < 1:
        raise ValueError(f"N={N} leaves no tail legs out of {n_legs}")

    U, S, Vh = truncated_svd(vec.reshape(2**N, -1), eps=eps, bdmax=bd_first,
                             renormalize=renormalize)
    A = [U.reshape(1, 2**N, len(S))]
    r = len(S)
    rest = torch.diag(S) @ Vh

    for _ in range(1, n_tail):
        U, S, Vh = truncated_svd(rest.reshape(r * 2, -1), eps=eps,
                                 bdmax=bd_remaining, renormalize=renormalize)
        A.append(U.reshape(r, 2, len(S)))
        r = len(S)
        rest = torch.diag(S) @ Vh

    A.append(rest.reshape(r, 2, 1))
    return from_tensors(A)


def field_to_hybrid(field: torch.Tensor, L: int, N: int, bd_first: int,
                    bd_remaining: int = None, eps: float = None,
                    renormalize: bool = False):
    """Compress a 2^L x 2^L field (row, col layout) into hybrid TT form."""
    _check_LN(L, N)
    if tuple(field.shape) != (2**L, 2**L):
        raise ValueError(f"field shape {tuple(field.shape)} != {(2**L, 2**L)}")
    return vector_to_hybrid(shuffle_axis(field, L).flatten(), 2 * L, N,
                            bd_first=bd_first, bd_remaining=bd_remaining,
                            eps=eps, renormalize=renormalize)


def hybrid_to_field(mps, L: int) -> torch.Tensor:
    """Contract a hybrid MPS back to a 2^L x 2^L field in (row, col) layout."""
    vec = contract(mps)
    return unshuffle_axis(vec.reshape(2**L, 2**L), L)


def contract(mps) -> torch.Tensor:
    """Dense vector of an MPS with arbitrary (possibly non-uniform) physical dims.

    ``MPS.contract`` works, but ``MPO.contract`` does not for hybrid shapes, so
    we keep a single explicit implementation for both.
    """
    A = tensors(mps)
    out = A[0]
    for T in A[1:]:
        l, p, r = out.shape
        _, q, s = T.shape
        out = torch.tensordot(out, T, dims=([2], [0])).reshape(l, p * q, s)
    return out.reshape(-1)


# --------------------------------------------------------------------------
# linear algebra with explicit truncation
# --------------------------------------------------------------------------
def _truncation_rank(S: np.ndarray, eps: float, bdmax: int) -> int:
    """Rank keeping discarded l2 weight below ``eps * ||S||``, then capped."""
    r = len(S)
    if eps is not None and r:
        # tail[i] = sum_{j >= i} S_j^2 : the l2 error incurred by keeping rank i
        tail = np.cumsum(S[::-1] ** 2)[::-1]
        tol = (eps * np.linalg.norm(S)) ** 2
        keep = np.nonzero(tail > tol)[0]
        r = int(keep[-1]) + 1 if len(keep) else 1
    if bdmax is not None:
        r = min(r, bdmax)
    return max(r, 1)


def _svd(M: np.ndarray):
    """SVD via LAPACK, falling back to the slower but sturdier driver.

    scipy/numpy are used rather than torch throughout the compression sweeps:
    measured on the matrix shapes these sweeps actually produce, torch's CPU
    LAPACK path is 3-7x slower (e.g. 8.1 ms vs 1.1 ms for a 256x128 QR), and
    the torch<->numpy conversions are zero-copy views costing ~1 us.
    """
    try:
        return sla.svd(M, full_matrices=False, check_finite=False)
    except np.linalg.LinAlgError:
        return sla.svd(M, full_matrices=False, check_finite=False,
                       lapack_driver="gesvd")


def truncated_svd(M, eps: float = None, bdmax: int = None,
                  renormalize: bool = False):
    """SVD truncated to relative l2 tolerance ``eps`` and/or rank ``bdmax``.

    ``eps`` is an *amplitude* tolerance: the discarded l2 weight is at most
    ``eps * ||S||``.  (``tt.svd`` instead treats ``eps`` as a squared-energy
    fraction, so its achievable amplitude error saturates near ``sqrt(eps)``.)

    ``renormalize`` rescales the retained singular values so the truncated
    matrix recovers the *full* Frobenius norm.  ``tt.svd`` does this by
    default; we do not, because under time stepping it would fake conservation
    of variance -- a truncation that loses norm must show up as lost norm.  The
    flag exists only to reproduce the legacy pipeline bit-for-bit.
    """
    Mn = M.numpy() if isinstance(M, torch.Tensor) else np.asarray(M)
    U, S, Vh = _svd(Mn)
    full_norm = np.linalg.norm(S)
    r = _truncation_rank(S, eps, bdmax)
    S = S[:r]
    if renormalize:
        kept = np.linalg.norm(S)
        if kept > 0:
            S = S * (full_norm / kept)
    return (torch.from_numpy(np.ascontiguousarray(U[:, :r])),
            torch.from_numpy(np.ascontiguousarray(S)),
            torch.from_numpy(np.ascontiguousarray(Vh[:r, :])))


def compress_cores(A: Sequence[torch.Tensor], caps: Sequence[int] = None,
                   eps: float = None) -> List[torch.Tensor]:
    """Two-sweep (QR then truncating SVD) compression of a list of 3-leg cores.

    ``caps[b]`` bounds bond ``b`` (between sites ``b`` and ``b+1``) individually,
    which is what the hybrid encoding needs and what ``MPS.compress`` -- a single
    scalar ``max_bd`` -- cannot express.  Shared with the MPO compressor.

    The sweeps run in numpy/LAPACK; see :func:`_svd` for why.
    """
    An = [(T.numpy() if isinstance(T, torch.Tensor) else np.asarray(T)).copy()
          for T in A]
    n = len(An)
    if n == 1:
        return [torch.from_numpy(np.ascontiguousarray(An[0]))]
    if caps is None:
        caps = [None] * (n - 1)
    if len(caps) != n - 1:
        raise ValueError(f"expected {n - 1} bond caps, got {len(caps)}")

    # sweep left -> right, QR, leaving sites 0..n-2 left-orthogonal
    for k in range(n - 1):
        l, p, r = An[k].shape
        Q, R = sla.qr(An[k].reshape(l * p, r), mode="economic", check_finite=False)
        An[k] = Q.reshape(l, p, -1)
        An[k + 1] = np.tensordot(R, An[k + 1], axes=([1], [0]))

    # sweep right -> left, SVD, truncating each bond
    for k in range(n - 1, 0, -1):
        l, p, r = An[k].shape
        U, S, Vh = _svd(An[k].reshape(l, p * r))
        rk = _truncation_rank(S, eps, caps[k - 1])
        An[k] = Vh[:rk].reshape(-1, p, r)
        # A[k-1] @ (U * S), written as a matmul: np.einsum with optimize=True
        # re-derives the contraction path on every call, which for a 3-operand
        # contraction this small costs more than the contraction itself.
        lm, pm, rm = An[k - 1].shape
        An[k - 1] = (An[k - 1].reshape(lm * pm, rm) @ (U[:, :rk] * S[:rk])
                     ).reshape(lm, pm, rk)

    return [torch.from_numpy(np.ascontiguousarray(x)) for x in An]


def hybrid_compress(mps, bd_first: int = None, bd_remaining: int = None,
                    eps: float = None):
    """Two-sweep compression with a *separate* cap on the head bond.

    ``MPS.compress`` accepts only a single scalar ``max_bd``; the hybrid
    encoding needs ``bd_first`` (head <-> tail) to differ from ``bd_remaining``.
    """
    A = tensors(mps)
    caps = [bd_first] + [bd_remaining] * (len(A) - 2)
    return from_tensors(compress_cores(A, caps=caps, eps=eps))


# --------------------------------------------------------------------------
# arithmetic (exact -- no implicit compression)
# --------------------------------------------------------------------------
def hybrid_add(x, y):
    """Exact sum. Bond dimensions add; caller decides when to compress."""
    A, B = tensors(x), tensors(y)
    if len(A) != len(B):
        raise ValueError("operands must have the same number of sites")
    n = len(A)
    out = []
    for k in range(n):
        a, b = A[k], B[k]
        if a.shape[1] != b.shape[1]:
            raise ValueError(f"physical dim mismatch at site {k}")
        if k == 0:
            out.append(torch.cat([a, b], dim=2))
        elif k == n - 1:
            out.append(torch.cat([a, b], dim=0))
        else:
            la, p, ra = a.shape
            lb, _, rb = b.shape
            top = torch.cat([a, a.new_zeros(la, p, rb)], dim=2)
            bot = torch.cat([b.new_zeros(lb, p, ra), b], dim=2)
            out.append(torch.cat([top, bot], dim=0))
    return from_tensors(out)


def hybrid_scale(x, c: float):
    """Exact scalar multiple (folded into the last core, bond dims unchanged)."""
    A = [T.clone() for T in tensors(x)]
    A[-1] = A[-1] * c
    return from_tensors(A)


def hybrid_sub(x, y):
    """Exact difference."""
    return hybrid_add(x, hybrid_scale(y, -1.0))


def hybrid_axpy(a: float, x, b: float, y):
    """Exact ``a*x + b*y``."""
    return hybrid_add(hybrid_scale(x, a), hybrid_scale(y, b))


def hybrid_constant(L: int, N: int, value: float = 1.0):
    """Uniform field as a rank-1 hybrid MPS."""
    _check_LN(L, N)
    dt = torch.get_default_dtype()
    A = [torch.ones(1, 2**N, 1, dtype=dt) * value]
    A += [torch.ones(1, 2, 1, dtype=dt) for _ in range(2 * L - N)]
    return from_tensors(A)


def project_mass(mps, L: int, N: int, target_mean: float):
    """Shift by a constant so the field mean equals ``target_mean``.

    Truncation does not preserve the mean, and for a passive scalar total mass
    is exactly conserved by both advection and diffusion, so restoring it is a
    physically justified correction.  Unlike rescaling the l2 norm this adds a
    rank-1 constant rather than amplifying the retained modes, so it does not
    move variance between scales.  Costs one bond.
    """
    current = hybrid_sum(mps) / 4**L
    return hybrid_add(mps, hybrid_constant(L, N, target_mean - current))


def hybrid_hadamard(x, y):
    """Exact elementwise product. Bond dimensions multiply."""
    A, B = tensors(x), tensors(y)
    if len(A) != len(B):
        raise ValueError("operands must have the same number of sites")
    out = []
    for k, (a, b) in enumerate(zip(A, B)):
        if a.shape[1] != b.shape[1]:
            raise ValueError(f"physical dim mismatch at site {k}")
        t = torch.einsum("lpr,upv->luprv", a, b)
        la, lb, p, ra, rb = t.shape
        out.append(t.reshape(la * lb, p, ra * rb))
    return from_tensors(out)


# --------------------------------------------------------------------------
# operator application
# --------------------------------------------------------------------------
def _apply_core(W: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """One standard site: MPO core ``W`` (l, in, r, out) onto MPS core ``T`` (a, in, b).

    Index convention verified numerically against dense references: axis 1 of an
    MPO core is the *input* (column) leg, axis 3 the *output* (row) leg.  The
    combined bond ordering (MPO bond slow, MPS bond fast) matches tensnet's own
    ``MPO.__matmul__``.
    """
    out = torch.einsum("lirj,aib->lajrb", W, T)
    l, a, j, r, b = out.shape
    return out.reshape(l * a, j, r * b)


def _apply_head(cores: Sequence[torch.Tensor], head: torch.Tensor) -> torch.Tensor:
    """Apply the first N MPO cores to the dense head without ever forming it densely.

    The fused head operator would have 4^N entries (4e9 already at N=16), so the
    head is temporarily un-fused into N binary legs, the cores are contracted one
    at a time, and the result is re-fused.  Cost O(2^N r^2 chi).

    Working layout of ``T`` is ``(a, b, *remaining_in, chi, *outs)`` with ``a``
    the MPS left bond and ``b`` the running MPO bond.
    """
    N = len(cores)
    a, pd, chi = head.shape
    if pd != 2**N:
        raise ValueError(f"head physical dim {pd} != 2^{N}")

    T = head.reshape(a, 1, *([2] * N), chi)
    for k, W in enumerate(cores):
        if W.shape[1] != 2 or W.shape[3] != 2:
            raise ValueError(f"head MPO core {k} must have physical dim 2")
        T = torch.tensordot(T, W, dims=([1, 2], [0, 1]))
        nd = T.dim()
        rem = N - k - 1
        perm = (
            [0, nd - 2]                       # a, new MPO bond
            + list(range(1, 1 + rem))         # remaining input legs
            + [1 + rem]                       # chi
            + list(range(2 + rem, nd - 2))    # outputs produced so far
            + [nd - 1]                        # this step's output leg
        )
        T = T.permute(perm)

    # T is (a, b, chi, out_0 .. out_{N-1}) -> (a, 2^N, b*chi)
    b, c = T.shape[1], T.shape[2]
    T = T.permute([0] + list(range(3, 3 + N)) + [1, 2])
    return T.reshape(a, 2**N, b * c)


def hybrid_apply(mpo, mps, N: int):
    """Apply a 2L-site MPO to a hybrid MPS with a 2^N head. Exact; no compression.

    Needed because ``MPO.__matmul__`` asserts equal site counts, which never
    holds for a hybrid MPS.
    """
    W = [node.tensor for node in mpo]
    A = tensors(mps)
    if len(W) != N + len(A) - 1:
        raise ValueError(
            f"MPO has {len(W)} sites but a hybrid MPS with {len(A)} sites and "
            f"N={N} needs {N + len(A) - 1}"
        )
    head = _apply_head(W[:N], A[0])
    tail = [_apply_core(w, t) for w, t in zip(W[N:], A[1:])]
    return from_tensors([head] + tail)


def hybrid_apply_compress(mpo, mps, N: int, bd_first: int = None,
                          bd_remaining: int = None, eps: float = None):
    """:func:`hybrid_apply` followed by :func:`hybrid_compress`."""
    return hybrid_compress(hybrid_apply(mpo, mps, N),
                           bd_first=bd_first, bd_remaining=bd_remaining, eps=eps)


# --------------------------------------------------------------------------
# diagnostics
# --------------------------------------------------------------------------
def hybrid_norm(mps) -> float:
    """l2 norm, computed by contraction of <psi|psi> (no dense vector formed)."""
    A = tensors(mps)
    E = torch.ones(1, 1, dtype=A[0].dtype)
    for T in A:
        E = torch.einsum("au,apr,upv->rv", E, T, T)
    return float(torch.sqrt(torch.clamp(E.reshape(-1)[0], min=0.0)))


def hybrid_sum(mps) -> float:
    """Sum of all entries (i.e. mass), by contracting against the all-ones vector."""
    A = tensors(mps)
    v = torch.ones(1, dtype=A[0].dtype)
    for T in A:
        v = torch.einsum("a,apr->r", v, T)
    return float(v.reshape(-1)[0])


def bond_dims(mps) -> List[int]:
    """Bond dimensions, head bond first."""
    return [T.shape[2] for T in tensors(mps)[:-1]]


def num_params(mps) -> int:
    """Parameter count of the hybrid representation."""
    return int(sum(T.numel() for T in tensors(mps)))


# --------------------------------------------------------------------------
# fused (dense-head) operator path
# --------------------------------------------------------------------------
def fuse_head_operator(mpo, N: int, sparse: bool = True):
    """Contract the first N MPO cores into an explicit 2^N x 2^N head operator.

    Returns a list of ``r`` matrices ``H[b]`` with ``H[b][O, I]``, one per value
    of the MPO bond leaving the head, so that

        (out)[O, b, c] = sum_I H[b][O, I] * A0[I, c].

    This is the alternative to the un-fusing path in :func:`_apply_head`: the
    head is treated as one dense linear-algebra object rather than as N binary
    legs.  It is built by iterated Kronecker products, one head leg at a time,
    so the full operator is never assembled from a dense contraction.

    ``sparse=True`` (default) returns ``scipy.sparse.csr_matrix``.  The dense
    form costs ``r * 4^N * 8`` bytes -- 0.62 GB at N=12 with bond 5 -- whereas
    these operators are banded permutation-like objects with O(r 2^N) nonzeros,
    so the sparse form is smaller by three to four orders of magnitude and is
    what makes N=12 practical.
    """
    import scipy.sparse as sp

    W = [node.tensor for node in mpo]
    if len(W) < N:
        raise ValueError(f"MPO has {len(W)} sites, need at least N={N}")
    cur = [sp.identity(1, format="csr", dtype=np.float64)]

    for k in range(N):
        Wk = W[k]
        l, pin, r, pout = Wk.shape
        if pin != 2 or pout != 2:
            raise ValueError(f"head MPO core {k} must have physical dim 2")
        if l != len(cur):
            raise ValueError(f"bond mismatch at head core {k}")
        nxt = []
        for b2 in range(r):
            acc = None
            for b1 in range(l):
                blk = Wk[b1, :, b2, :].numpy().T        # (out, in)
                if not blk.any():
                    continue
                term = sp.kron(cur[b1], sp.csr_matrix(blk), format="csr")
                acc = term if acc is None else (acc + term)
            if acc is None:
                acc = sp.csr_matrix((2 ** (k + 1), 2 ** (k + 1)), dtype=np.float64)
            nxt.append(acc.tocsr())
        cur = nxt

    if sparse:
        return cur
    return [np.asarray(m.todense()) for m in cur]


def head_operator_nnz(head_op) -> int:
    """Total stored nonzeros of a fused head operator."""
    return int(sum(m.nnz if hasattr(m, "nnz") else np.count_nonzero(m) for m in head_op))


def hybrid_apply_fused(mpo, mps, N: int, head_op=None):
    """Apply an MPO using an explicit fused head operator (no un-fusing).

    Numerically identical to :func:`hybrid_apply`; the head is handled by
    ``r`` sparse mat-mats of shape (2^N, 2^N) x (2^N, chi) instead of by
    contracting N binary cores.  Pass a precomputed ``head_op`` to reuse it
    across time steps, which is the point: it is built once and applied
    thousands of times.
    """
    W = [node.tensor for node in mpo]
    A = tensors(mps)
    if len(W) != N + len(A) - 1:
        raise ValueError(
            f"MPO has {len(W)} sites but a hybrid MPS with {len(A)} sites and "
            f"N={N} needs {N + len(A) - 1}")
    if head_op is None:
        head_op = fuse_head_operator(mpo, N)

    A0 = A[0]
    if A0.shape[0] != 1:
        raise ValueError("head must have left bond 1")
    chi = A0.shape[2]
    X = A0[0].numpy()                                  # (2^N, chi)
    r = len(head_op)
    out = np.empty((2**N, r, chi), dtype=np.float64)
    for b in range(r):
        out[:, b, :] = head_op[b] @ X                  # sparse mat-mat
    head = torch.from_numpy(out.reshape(1, 2**N, r * chi))

    tail = [_apply_core(w, t) for w, t in zip(W[N:], A[1:])]
    return from_tensors([head] + tail)


# --------------------------------------------------------------------------
# pointwise evaluation (no dense field)
# --------------------------------------------------------------------------
def hybrid_eval_at(mps, L: int, rows, cols) -> torch.Tensor:
    """Evaluate a hybrid MPS at given (row, col) pairs without contracting it.

    Essential above L ~ 10, where the dense field cannot be formed at all: at
    L=16 it would be 2^32 cells.  Vectorised over the query points.
    """
    A = tensors(mps)
    N = int(round(np.log2(A[0].shape[1])))
    rows = np.asarray(rows, dtype=np.int64)
    cols = np.asarray(cols, dtype=np.int64)
    npts = len(rows)

    bits = np.empty((npts, 2 * L), dtype=np.int64)
    for j in range(L):
        bits[:, 2 * j] = (rows >> (L - 1 - j)) & 1        # even legs: row
        bits[:, 2 * j + 1] = (cols >> (L - 1 - j)) & 1    # odd legs: col

    head_idx = np.zeros(npts, dtype=np.int64)
    for j in range(N):
        head_idx = head_idx * 2 + bits[:, j]

    v = A[0][0][torch.from_numpy(head_idx)]               # (npts, chi)
    for k, T in enumerate(A[1:]):
        b = torch.from_numpy(bits[:, N + k])
        v = torch.einsum("pa,par->pr", v, T[:, b, :].permute(1, 0, 2))
    return v[:, 0]
