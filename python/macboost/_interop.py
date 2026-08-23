"""Array interop: accept MLX arrays, PyTorch tensors, and anything else
exposing the buffer protocol or DLPack, and hand results back in the
caller's array family.

MLX arrays live in unified memory, so `np.asarray(mx_array)` is a
zero-copy view (buffer protocol) — an MLX embedding pipeline can feed a
booster without the data ever moving. `np.from_dlpack` is the second
route (CPU DLPack producers); `.cpu()` is the last resort for device
tensors (e.g. PyTorch MPS) that expose neither to numpy.
"""

import numpy as np


def to_numpy(X):
    """Any array-like -> numpy array, zero-copy when the source allows."""
    if X is None or isinstance(X, np.ndarray):
        return X
    try:
        return np.asarray(X)           # buffer protocol / __array__ / lists
    except Exception:
        pass
    if hasattr(X, "__dlpack__"):
        try:
            return np.from_dlpack(X)
        except Exception:
            pass
    if hasattr(X, "cpu"):              # device tensors: torch MPS/CUDA
        return to_numpy(X.cpu())
    if hasattr(X, "numpy"):
        return np.asarray(X.numpy())
    raise TypeError(f"cannot convert {type(X).__name__} to a numpy array")


def wrapper_for(X):
    """Return f(np.ndarray) -> same array family as X (identity for numpy,
    lists, pandas, etc.). Only MLX and PyTorch are wrapped back; everything
    else gets numpy, as before."""
    mod = type(X).__module__ or ""
    if mod.startswith("mlx."):
        import mlx.core as mx
        return lambda a: mx.array(np.ascontiguousarray(a))
    if mod.startswith("torch"):
        import torch
        return lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(X.device)
    return lambda a: a
