"""Affine input projection stored inside the model (PCA, scalers, N of them)."""

import numpy as np


class Projection:
    """Blocks of `matrix @ (x[inputs] - mean)` plus passthrough columns.
    Output column order: each block's outputs in order, then passthrough.
    `fill` (one value per raw column, NaN = leave) replaces NaN first."""

    def __init__(self, input_width, blocks=(), passthrough=(), fill=None):
        self.input_width = int(input_width)
        self.blocks = []
        for b in blocks:
            self.add_block(b["inputs"], b["mean"], b["matrix"])
        self.passthrough = [int(i) for i in passthrough]
        self.fill = None if fill is None else np.asarray(fill, dtype=np.float32).ravel()

    @classmethod
    def from_pca(cls, components, mean, input_width=None, inputs=None, passthrough=()):
        """From a fitted PCA: `components` is (k, d) (sklearn's components_),
        `mean` is (d,). `inputs` are the raw columns the PCA was fit on
        (default: the first d)."""
        components = np.asarray(components, dtype=np.float32)
        d = components.shape[1]
        inputs = list(range(d)) if inputs is None else [int(i) for i in inputs]
        width = input_width if input_width is not None else max(inputs + list(passthrough)) + 1
        p = cls(width, passthrough=passthrough)
        p.add_block(inputs, mean, components)
        return p

    def add_block(self, inputs, mean, matrix):
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[1] != len(inputs):
            raise ValueError(f"matrix must be (outputs, {len(inputs)}), got {matrix.shape}")
        self.blocks.append({"inputs": [int(i) for i in inputs],
                            "mean": np.asarray(mean, dtype=np.float32).ravel(),
                            "matrix": matrix})
        return self

    @property
    def output_width(self):
        return sum(b["matrix"].shape[0] for b in self.blocks) + len(self.passthrough)

    def apply(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.input_width:
            raise ValueError(f"X must be (n, {self.input_width}), got {X.shape}")
        if self.fill is not None:
            X = np.where(np.isnan(X) & ~np.isnan(self.fill), self.fill, X)
        parts = [(X[:, b["inputs"]] - b["mean"]) @ b["matrix"].T for b in self.blocks]
        if self.passthrough:
            parts.append(X[:, self.passthrough])
        return np.concatenate(parts, axis=1).astype(np.float32)

    def to_dict(self):
        d = {"inputWidth": self.input_width,
             "blocks": [{"inputs": b["inputs"], "mean": b["mean"].tolist(),
                         "matrix": b["matrix"].ravel().tolist(),
                         "outputs": int(b["matrix"].shape[0])} for b in self.blocks],
             "passthrough": self.passthrough}
        if self.fill is not None:
            d["fill"] = [None if np.isnan(v) else float(v) for v in self.fill]
        return d

    @classmethod
    def from_dict(cls, d):
        if d is None:
            return None
        p = cls(d["inputWidth"], passthrough=d.get("passthrough") or [])
        for b in d["blocks"]:
            p.add_block(b["inputs"], b["mean"],
                        np.asarray(b["matrix"], dtype=np.float32).reshape(b["outputs"], -1))
        if d.get("fill") is not None:
            p.fill = np.asarray([np.nan if v is None else v for v in d["fill"]], dtype=np.float32)
        return p

    @staticmethod
    def coerce(p):
        if p is None or isinstance(p, Projection):
            return p
        if isinstance(p, dict):
            return Projection.from_dict(p)
        raise TypeError("projection must be a macboost.Projection or its dict form")
