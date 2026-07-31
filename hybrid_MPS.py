import torch
import sys
import config as cfg
sys.path.append(cfg.TENSNET_SRC)   # external tensnet library (see config.py)
import tensnet as tt

def vector_to_hybrid_mps(
    vec: torch.Tensor,
    L: int,
    N: int,
    bd_first: int,
    bd_remaining: int = None,
    eps: float = None,
    naive_svd: bool = True,
) -> tt.MPS:
    """
    Converts a state vector into a hybrid MPS format with a larger physical dimension
    for the first tensor and standard binary dimensions for the remaining tensors.
    
    Args:
        vec (torch.Tensor): State vector of size 2^L to decompose.
        L (int): Total number of qubits/physical legs.
        N (int): Number of qubits grouped in the first tensor (first tensor has physical dim 2^N).
        bd_first (int): Bond dimension at the first bond (between first and second tensor).
        bd_remaining (int, optional): Maximum bond dimension for remaining bonds. 
                                       If None, uses bd_first. Defaults to None.
        eps (float, optional): SVD truncation threshold in l2 norm. Defaults to None.
        naive_svd (bool, optional): If True, uses naive SVD approach. 
                                     If False, uses QR+SVD. Defaults to True.
    
    Returns:
        MPS: Hybrid MPS with first tensor having physical dimension 2^N and 
             remaining tensors having physical dimension 2.
    
    Raises:
        AssertionError: If vector size doesn't match 2^L or if N >= L.
    
    Example:
        >>> vec = torch.randn(2**10)  # 10 qubits
        >>> hybrid_mps = vector_to_hybrid_mps(vec, L=10, N=4, bd_first=10, bd_remaining=2)
        >>> # First tensor: shape (1, 2^4=16, 10)
        >>> # Remaining 6 tensors: shape varies but physical dim is 2
    """
    assert torch.numel(vec) == 2**L, f"Vector size {torch.numel(vec)} doesn't match 2^{L} = {2**L}"
    assert N < L, f"N={N} must be less than L={L}"
    assert N > 0, f"N={N} must be positive"
    
    if bd_remaining is None:
        bd_remaining = bd_first
    
    temp = torch.clone(vec)
    d_first = 2**N  # Physical dimension of first tensor
    d_std = 2       # Physical dimension of remaining tensors
    L_remaining = L - N  # Number of remaining standard tensors
    
    if naive_svd:
        A = []
        
        # First decomposition: separate first N qubits from the rest
        # Reshape: (2^N, 2^(L-N))
        U, S, V = tt.svd(
            temp.reshape([d_first, 2**(L - N)]), 
            eps=eps, 
            bdmax=bd_first
        )
        
        # First tensor with physical dimension 2^N
        A.append(U.reshape(1, d_first, len(S)))
        
        # Continue with standard binary decomposition for remaining qubits
        temp = torch.diag(S.type(temp.dtype)) @ V
        
        # Decompose remaining L-N qubits with physical dimension 2
        for i in range(1, L_remaining):
            U, S, V = tt.svd(
                temp.reshape([d_std * len(S), 2**(L_remaining - i)]), 
                eps=eps, 
                bdmax=bd_remaining
            )
            A.append(U.reshape(-1, d_std, len(S)))
            temp = torch.diag(S.type(temp.dtype)) @ V
        
        # Last tensor
        A.append(temp.reshape(len(S), d_std, 1))
        
        return tt.list_to_mps(A)
    
    else:
        # QR-based decomposition (right-to-left)
        A = []
        W = temp.reshape(-1, d_std)
        
        # Process remaining standard tensors (right to left)
        for i in range(L_remaining - 1, 0, -1):
            Q, R = tt.qr(W)
            U, S, V = tt.svd(R, eps, bd_remaining)
            A.append(V.reshape(V.shape[0], d_std, -1))
            W = (W @ V.T).reshape(-1, d_std * A[-1].shape[0])
        
        # Handle the transition tensor (connects first hybrid tensor to standard ones)
        Q, R = tt.qr(W)
        U, S, V = tt.svd(R, eps, bd_first)
        A.append(V.reshape(V.shape[0], d_std, -1))
        
        # First hybrid tensor with physical dimension 2^N
        W = (W @ V.T).reshape(1, d_first, -1)
        A.append(W)
        
        return tt.list_to_mps(A[::-1])


def vector_to_hybrid_mps_p4(
    vec: torch.Tensor,
    L: int,
    N: int,
    bd_first: int,
    bd_remaining: int = None,
    eps: float = None,
    naive_svd: bool = True,
) -> tt.MPS:
    """
    Converts a state vector into a hybrid MPS format with a larger physical dimension
    for the first tensor and physical dimension 4 for the remaining tensors.

    Each remaining tensor groups 2 qubits (physical dim 4 = 2^2), so (L - N) must
    be even. The number of remaining tensors is M = (L - N) // 2.

    Args:
        vec (torch.Tensor): State vector of size 2^L to decompose.
        L (int): Total number of qubits/physical legs (must satisfy (L - N) % 2 == 0).
        N (int): Number of qubits grouped in the first tensor (physical dim 2^N).
        bd_first (int): Bond dimension at the first bond (between first and second tensor).
        bd_remaining (int, optional): Maximum bond dimension for remaining bonds.
                                       If None, uses bd_first. Defaults to None.
        eps (float, optional): SVD truncation threshold in l2 norm. Defaults to None.
        naive_svd (bool, optional): If True, uses naive SVD approach.
                                     If False, uses QR+SVD. Defaults to True.

    Returns:
        MPS: Hybrid MPS with first tensor having physical dimension 2^N and
             remaining tensors having physical dimension 4.

    Raises:
        AssertionError: If vector size doesn't match 2^L, N >= L, or (L - N) is odd.

    Example:
        >>> vec = torch.randn(2**24)  # 24 qubits
        >>> hybrid_mps = vector_to_hybrid_mps_p4(vec, L=24, N=12, bd_first=100)
        >>> # First tensor:      shape (1, 2^12=4096, bd_first)
        >>> # Remaining 6 tensors: physical dim 4, bond dims up to bd_first
    """
    assert torch.numel(vec) == 2**L, f"Vector size {torch.numel(vec)} doesn't match 2^{L} = {2**L}"
    assert N < L, f"N={N} must be less than L={L}"
    assert N > 0, f"N={N} must be positive"
    assert (L - N) % 2 == 0, f"(L - N) = {L - N} must be even so remaining qubits can be grouped into p=4 tensors"

    if bd_remaining is None:
        bd_remaining = bd_first

    temp = torch.clone(vec)
    d_first = 2**N  # Physical dimension of first tensor
    d_std = 4       # Physical dimension of remaining tensors (2 qubits each)
    M = (L - N) // 2  # Number of remaining tensors with p=4

    if naive_svd:
        A = []

        # First decomposition: separate first N qubits from the rest
        # Reshape: (2^N, 4^M)
        U, S, V = tt.svd(
            temp.reshape([d_first, 4**M]),
            eps=eps,
            bdmax=bd_first
        )

        # First tensor with physical dimension 2^N
        A.append(U.reshape(1, d_first, len(S)))

        # Continue with p=4 decomposition for remaining qubit pairs
        temp = torch.diag(S.type(temp.dtype)) @ V

        for i in range(1, M):
            U, S, V = tt.svd(
                temp.reshape([d_std * len(S), 4**(M - i)]),
                eps=eps,
                bdmax=bd_remaining
            )
            A.append(U.reshape(-1, d_std, len(S)))
            temp = torch.diag(S.type(temp.dtype)) @ V

        # Last tensor
        A.append(temp.reshape(len(S), d_std, 1))

        return tt.list_to_mps(A)

    else:
        # QR-based decomposition (right-to-left)
        A = []
        W = temp.reshape(-1, d_std)

        # Process remaining p=4 tensors (right to left)
        for i in range(M - 1, 0, -1):
            Q, R = tt.qr(W)
            U, S, V = tt.svd(R, eps, bd_remaining)
            A.append(V.reshape(V.shape[0], d_std, -1))
            W = (W @ V.T).reshape(-1, d_std * A[-1].shape[0])

        # Handle the transition tensor (connects first hybrid tensor to p=4 ones)
        Q, R = tt.qr(W)
        U, S, V = tt.svd(R, eps, bd_first)
        A.append(V.reshape(V.shape[0], d_std, -1))

        # First hybrid tensor with physical dimension 2^N
        W = (W @ V.T).reshape(1, d_first, -1)
        A.append(W)

        return tt.list_to_mps(A[::-1])


def hybrid_mps_shapes(L: int, N: int, bd_first: int, bd_remaining: int = None) -> list[tuple]:
    """
    Calculate the expected shapes of tensors in a hybrid MPS.
    
    Args:
        L (int): Total number of qubits.
        N (int): Number of qubits in first tensor.
        bd_first (int): Bond dimension at first bond.
        bd_remaining (int, optional): Max bond dimension for remaining bonds.
    
    Returns:
        List[tuple]: List of tensor shapes (left_bond, physical_dim, right_bond).
    
    Example:
        >>> shapes = hybrid_mps_shapes(L=10, N=4, bd_first=10, bd_remaining=2)
        >>> print(shapes[0])  # First tensor
        (1, 16, 10)
        >>> print(shapes[1])  # Second tensor
        (10, 2, 2)
    """
    if bd_remaining is None:
        bd_remaining = bd_first
    
    d_first = 2**N
    d_std = 2
    L_remaining = L - N
    
    shapes = []
    
    # First tensor
    shapes.append((1, d_first, bd_first))
    
    # Remaining tensors with standard bond dimension logic
    left_bd = bd_first
    for i in range(L_remaining - 1):
        i_sym = min(i + 1, L_remaining - i - 1)
        right_bd = min(d_std**i_sym, bd_remaining, left_bd * d_std)
        shapes.append((left_bd, d_std, right_bd))
        left_bd = right_bd
    
    # Last tensor
    shapes.append((left_bd, d_std, 1))
    
    return shapes