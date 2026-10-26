"""Fitters for calibrations that fold into the trees exactly."""

import numpy as np


def fit_platt(margin, y, iters=50):
    """Platt scaling: (A, B) minimising the log loss of sigmoid(A*s + B)."""
    s = np.asarray(margin, dtype=np.float64).ravel()
    t = np.asarray(y, dtype=np.float64).ravel()
    if s.shape != t.shape:
        raise ValueError("margin and y must have the same length")
    if not (0 < t.mean() < 1):
        raise ValueError("calibration data needs both classes")
    a, b = 1.0, 0.0
    nll = _platt_nll(s, t, a, b)
    for _ in range(iters):
        z = a * s + b
        p = 1.0 / (1.0 + np.exp(-z))
        g = p - t
        w = p * (1.0 - p) + 1e-12
        grad = np.array([np.dot(g, s), g.sum()])
        hess = np.array([[np.dot(w, s * s), np.dot(w, s)], [np.dot(w, s), w.sum()]])
        step = np.linalg.solve(hess + 1e-9 * np.eye(2), grad)
        lr = 1.0
        while lr > 1e-4:                                  # backtracking
            na, nb = a - lr * step[0], b - lr * step[1]
            new = _platt_nll(s, t, na, nb)
            if new <= nll:
                break
            lr *= 0.5
        if abs(nll - new) < 1e-12 * max(1.0, nll):
            a, b, nll = na, nb, new
            break
        a, b, nll = na, nb, new
    if not a > 0:
        raise ValueError(f"Platt fit gave a non-positive slope ({a:.4g}); "
                         "the margin does not rank the labels")
    return float(a), float(b)


def _platt_nll(s, t, a, b):
    z = a * s + b
    return float(np.sum(np.logaddexp(0.0, -z) * t + np.logaddexp(0.0, z) * (1 - t)))


def fit_temperature(margin, y, lo=-6.0, hi=6.0, iters=80):
    """Temperature scaling: T > 0 minimising the softmax log loss of s / T.
    Golden-section search on log T."""
    s = np.asarray(margin, dtype=np.float64)
    t = np.asarray(y).ravel().astype(np.int64)
    if s.ndim != 2 or s.shape[0] != t.shape[0]:
        raise ValueError("margin must be (rows, classes) matching y")
    rows = np.arange(s.shape[0])

    def nll(u):
        z = s * np.exp(-u)
        m = z.max(axis=1)
        return float(np.sum(m + np.log(np.exp(z - m[:, None]).sum(axis=1)) - z[rows, t]))

    phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c, d = b - phi * (b - a), a + phi * (b - a)
    fc, fd = nll(c), nll(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - phi * (b - a); fc = nll(c)
        else:
            a, c, fc = c, d, fd
            d = a + phi * (b - a); fd = nll(d)
    return float(np.exp((a + b) / 2.0))
