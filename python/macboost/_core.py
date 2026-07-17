"""ctypes bridge to libmacboostc.dylib (the Swift core's C ABI)."""

import ctypes
import json
import os
from pathlib import Path

import numpy as np


def _find_library() -> str:
    candidates = []
    if env := os.environ.get("MACBOOST_LIB"):
        candidates.append(Path(env))
    pkg_dir = Path(__file__).resolve().parent
    candidates.append(pkg_dir / "libmacboostc.dylib")
    # Running from a source checkout: ../../.build/release relative to python/macboost.
    repo = pkg_dir.parent.parent
    candidates.append(repo / ".build" / "release" / "libmacboostc.dylib")
    for c in candidates:
        if c.is_file():
            return str(c)
    raise OSError(
        "libmacboostc.dylib not found. Build it with `swift build -c release` "
        "(or scripts/build_python.sh to bundle it), or set MACBOOST_LIB."
    )


_c_char_pp = ctypes.POINTER(ctypes.c_char_p)
_f32_p = ctypes.POINTER(ctypes.c_float)

_lib_instance = None


class _LazyLib:
    """Defers dylib loading so `import macboost` works on any platform;
    only training/native inference require the Metal core."""

    def __getattr__(self, name):
        global _lib_instance
        if _lib_instance is None:
            if os.environ.get("MACBOOST_FORCE_PYTHON"):
                raise OSError("MACBOOST_FORCE_PYTHON is set")
            _lib_instance = ctypes.CDLL(_find_library())
            _configure(_lib_instance)
        return getattr(_lib_instance, name)


def native_available() -> bool:
    try:
        _lib.macboost_num_trees   # noqa: B018 — probe the load
        return True
    except OSError:
        return False


_lib = _LazyLib()

def _configure(_lib):
    _lib.macboost_train.restype = ctypes.c_void_p
    _lib.macboost_train.argtypes = [
        ctypes.c_char_p, _f32_p, ctypes.c_int64, ctypes.c_int64, ctypes.c_int32,
        _f32_p, _f32_p, _f32_p, ctypes.c_int64, _f32_p, _c_char_pp,
    ]
    _lib.macboost_select_features.restype = ctypes.c_int32
    _lib.macboost_select_features.argtypes = [
        ctypes.c_char_p, _f32_p, ctypes.c_int64, ctypes.c_int64, _f32_p, _f32_p,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_float, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_int32), _f32_p,
        _c_char_pp,
    ]
    _lib.macboost_predict_contrib.restype = ctypes.c_int32
    _lib.macboost_predict_contrib.argtypes = [
        ctypes.c_void_p, _f32_p, ctypes.c_int64, ctypes.c_int64, _f32_p, _c_char_pp,
    ]
    _lib.macboost_feature_importance.restype = ctypes.c_int32
    _lib.macboost_feature_importance.argtypes = [ctypes.c_void_p, ctypes.c_int32, _f32_p]
    _lib.macboost_predict.restype = ctypes.c_int32
    _lib.macboost_predict.argtypes = [
        ctypes.c_void_p, _f32_p, ctypes.c_int64, ctypes.c_int64, _f32_p, _c_char_pp,
    ]
    _lib.macboost_save.restype = ctypes.c_int32
    _lib.macboost_save.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _c_char_pp]
    _lib.macboost_load.restype = ctypes.c_void_p
    _lib.macboost_load.argtypes = [ctypes.c_char_p, _c_char_pp]
    for name in ("macboost_num_trees", "macboost_num_features", "macboost_best_iteration",
                 "macboost_num_classes"):
        fn = getattr(_lib, name)
        fn.restype = ctypes.c_int64
        fn.argtypes = [ctypes.c_void_p]
    _lib.macboost_warnings.restype = ctypes.c_void_p   # char* we must free ourselves
    _lib.macboost_warnings.argtypes = [ctypes.c_void_p]
    _lib.macboost_free.restype = None
    _lib.macboost_free.argtypes = [ctypes.c_void_p]
    _lib.macboost_free_string.restype = None
    _lib.macboost_free_string.argtypes = [ctypes.c_char_p]


class MacBoostError(RuntimeError):
    pass


def _take_error(err: ctypes.c_char_p) -> str:
    msg = err.value.decode() if err.value else "unknown error"
    # err.value copies the bytes; free the C string.
    _lib.macboost_free_string(err)
    return msg


def _feature_major(X) -> tuple[np.ndarray, int, int]:
    """(n, f) array-like -> contiguous feature-major float32 + (rows, cols)."""
    a = np.asarray(X, dtype=np.float32)
    if a.ndim != 2:
        raise ValueError(f"X must be 2-dimensional, got shape {a.shape}")
    rows, cols = a.shape
    return np.ascontiguousarray(a.T), rows, cols


def _f32(a: np.ndarray) -> "ctypes._Pointer":
    return a.ctypes.data_as(_f32_p)


class _Handle:
    """Owns a booster handle and frees it on GC."""

    def __init__(self, ptr: int):
        self.ptr = ptr

    def __del__(self):
        if getattr(self, "ptr", None):
            _lib.macboost_free(self.ptr)
            self.ptr = None


def train(config: dict, X, y, eval_set=None, sample_weight=None) -> _Handle:
    # Row-major fast path: numpy's native layout goes straight through the
    # ABI with no transpose copy (the core strides over it during binning).
    # Warm starts still need the feature-major layout for seed predictions.
    if "init_model" in config:
        Xf, rows, cols = _feature_major(X)
        layout = 0
    else:
        a = np.asarray(X, dtype=np.float32)
        if a.ndim != 2:
            raise ValueError(f"X must be 2-dimensional, got shape {a.shape}")
        Xf = np.ascontiguousarray(a)          # no-op if already C-contiguous f32
        rows, cols = a.shape
        layout = 1
    ya = np.ascontiguousarray(np.asarray(y, dtype=np.float32).ravel())
    if ya.shape[0] != rows:
        raise ValueError(f"y has {ya.shape[0]} rows, X has {rows}")
    w_ptr = None
    if sample_weight is not None:
        wa = np.ascontiguousarray(np.asarray(sample_weight, dtype=np.float32).ravel())
        if wa.shape[0] != rows:
            raise ValueError(f"sample_weight has {wa.shape[0]} rows, X has {rows}")
        w_ptr = _f32(wa)

    xv_ptr, yv_ptr, vrows = None, None, 0
    if eval_set is not None:
        Xv, yv = eval_set
        Xvf, vrows, vcols = _feature_major(Xv)
        if vcols != cols:
            raise ValueError("eval_set feature count differs from X")
        yva = np.ascontiguousarray(np.asarray(yv, dtype=np.float32).ravel())
        if yva.shape[0] != vrows:
            raise ValueError("eval_set X/y row mismatch")
        xv_ptr, yv_ptr = _f32(Xvf), _f32(yva)

    err = ctypes.c_char_p()
    ptr = _lib.macboost_train(
        json.dumps(config).encode(), _f32(Xf), rows, cols, layout, _f32(ya), w_ptr,
        xv_ptr, vrows, yv_ptr, ctypes.byref(err),
    )
    if not ptr:
        raise MacBoostError(_take_error(err))
    return _Handle(ptr)


def select_features(config: dict, X, y, sample_weight=None,
                    rounds: int = 20, trees: int = 0, alpha: float = 0.05,
                    seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boruta shadow selection in the native core. Returns (hits, decision,
    gain_ratio) where decision is 2=confirmed, 1=tentative, 0=rejected."""
    Xf, rows, cols = _feature_major(X)
    ya = np.ascontiguousarray(np.asarray(y, dtype=np.float32).ravel())
    if ya.shape[0] != rows:
        raise ValueError(f"y has {ya.shape[0]} rows, X has {rows}")
    w_ptr = None
    if sample_weight is not None:
        wa = np.ascontiguousarray(np.asarray(sample_weight, dtype=np.float32).ravel())
        if wa.shape[0] != rows:
            raise ValueError(f"sample_weight has {wa.shape[0]} rows, X has {rows}")
        w_ptr = _f32(wa)
    hits = np.empty(cols, dtype=np.int32)
    decision = np.empty(cols, dtype=np.int32)
    ratio = np.empty(cols, dtype=np.float32)
    err = ctypes.c_char_p()
    rc = _lib.macboost_select_features(
        json.dumps(config).encode(), _f32(Xf), rows, cols, _f32(ya), w_ptr,
        rounds, trees, alpha, seed,
        hits.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        decision.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
        _f32(ratio), ctypes.byref(err))
    if rc != 0:
        raise MacBoostError(_take_error(err))
    return hits, decision, ratio


def predict(handle: _Handle, X) -> np.ndarray:
    Xf, rows, cols = _feature_major(X)
    k = _lib.macboost_num_classes(handle.ptr)
    out = np.empty(rows * k, dtype=np.float32)
    err = ctypes.c_char_p()
    rc = _lib.macboost_predict(handle.ptr, _f32(Xf), rows, cols, _f32(out),
                               ctypes.byref(err))
    if rc != 0:
        raise MacBoostError(_take_error(err))
    return out.reshape(rows, k) if k > 1 else out


def predict_contrib(handle: _Handle, X) -> np.ndarray:
    Xf, rows, cols = _feature_major(X)
    k = _lib.macboost_num_classes(handle.ptr)
    out = np.empty(rows * k * (cols + 1), dtype=np.float32)
    err = ctypes.c_char_p()
    rc = _lib.macboost_predict_contrib(handle.ptr, _f32(Xf), rows, cols,
                                       _f32(out), ctypes.byref(err))
    if rc != 0:
        raise MacBoostError(_take_error(err))
    return out.reshape(rows, k * (cols + 1))


def feature_importance(handle: _Handle, importance_type: str) -> np.ndarray:
    n = _lib.macboost_num_features(handle.ptr)
    out = np.empty(n, dtype=np.float32)
    t = 1 if importance_type == "split" else 0
    if _lib.macboost_feature_importance(handle.ptr, t, _f32(out)) != 0:
        raise MacBoostError("feature_importance failed")
    return out


def save(handle: _Handle, path: str) -> None:
    err = ctypes.c_char_p()
    if _lib.macboost_save(handle.ptr, str(path).encode(), ctypes.byref(err)) != 0:
        raise MacBoostError(_take_error(err))


def load(path: str) -> _Handle:
    err = ctypes.c_char_p()
    ptr = _lib.macboost_load(str(path).encode(), ctypes.byref(err))
    if not ptr:
        raise MacBoostError(_take_error(err))
    return _Handle(ptr)


def warnings_of(handle: _Handle) -> list[str]:
    ptr = _lib.macboost_warnings(handle.ptr)
    if not ptr:
        return []
    text = ctypes.string_at(ptr).decode()
    _lib.macboost_free_string(ctypes.cast(ptr, ctypes.c_char_p))
    return text.splitlines()


def num_trees(handle: _Handle) -> int:
    return _lib.macboost_num_trees(handle.ptr)


def num_features(handle: _Handle) -> int:
    return _lib.macboost_num_features(handle.ptr)


def best_iteration(handle: _Handle) -> int:
    return _lib.macboost_best_iteration(handle.ptr)
