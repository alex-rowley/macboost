"""Pure-Python (numpy) inference over MacBoost model.json files.

Training requires the Metal core (Apple silicon), but a trained model is
just trees — this backend scores them anywhere Python runs, so models
trained on a Mac deploy to Linux with nothing but `pip install macboost`.
Semantics match the native predictor exactly: `x <= threshold` goes left,
NaN follows the learned default direction, categorical splits test bitmask
membership (unseen/out-of-range categories route like missing), and
poisson/tweedie outputs are exp-transformed.

Scoring is level-synchronous over the whole forest: trees are packed into
one flat node arena and a (rows, trees) position matrix advances every
row through every tree simultaneously, so a predict call costs
O(max_depth) numpy operations rather than O(trees * depth) — the
difference between milliseconds and microseconds per call once numpy op
overhead, not arithmetic, is the bottleneck.
"""

import json

import numpy as np

_LOG_LINK_OBJECTIVES = {5, 6}          # poisson, tweedie


class PyModel:
    """Reads the JSON written by MacBooster.save / save_model."""

    def __init__(self, path):
        with open(path) as fh:
            m = json.load(fh)
        if m.get("version") != 1:
            raise ValueError("unsupported model file version")
        self.objective = int(m["objective"])
        self.num_bins = int(m["numBins"])
        self.base_score = np.float32(m["baseScore"])
        self.num_features = int(m["numFeatures"])
        self.num_classes = int(m.get("numClasses") or 1)
        self.base_scores = np.asarray(
            m.get("baseScores") or [m["baseScore"]], dtype=np.float32)
        self.feature_names = m.get("featureNames")
        self._raw = m
        self.trees = []
        for t in m["trees"]:
            tree = {
                "feature": np.asarray(t["feature"], dtype=np.int64),
                "threshold": np.asarray(t["threshold"], dtype=np.float32),
                "leaf": np.asarray(t["leaf"], dtype=np.float32),
                "flags": np.asarray(t["flags"], dtype=np.uint8),
                "gain": np.asarray(t.get("gain") or [], dtype=np.float32),
            }
            mask = np.asarray(t.get("catMask") or [], dtype=np.uint64)
            tree["catMask"] = mask.reshape(-1, 8) if mask.size else None
            self.trees.append(tree)
        self._packed = None

    @property
    def num_trees(self):
        return len(self.trees)

    def _pack(self):
        """Flat forest arena: node arrays of every tree concatenated, plus
        per-tree offsets. Built once, on first predict."""
        if self._packed is None:
            sizes = np.array([t["feature"].size for t in self.trees],
                             dtype=np.int64)
            off = np.zeros(len(self.trees), dtype=np.int64)
            np.cumsum(sizes[:-1], out=off[1:])
            depth = 0
            for s in sizes:
                d = 0
                while (1 << (d + 1)) - 1 < s:
                    d += 1
                depth = max(depth, d)
            cat = None
            if any(t["catMask"] is not None for t in self.trees):
                cat = np.concatenate([
                    t["catMask"] if t["catMask"] is not None
                    else np.zeros((t["feature"].size, 8), dtype=np.uint64)
                    for t in self.trees])
            self._packed = {
                "off": off,
                "feature": np.concatenate([t["feature"] for t in self.trees]),
                "threshold": np.concatenate([t["threshold"] for t in self.trees]),
                "leaf": np.concatenate([t["leaf"] for t in self.trees]),
                "flags": np.concatenate([t["flags"] for t in self.trees]),
                "depth": depth,
                "catMask": cat,
            }
        return self._packed

    def predict_raw(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.num_features:
            raise ValueError(
                f"X must be (n, {self.num_features}), got {X.shape}")
        n = X.shape[0]
        K = self.num_classes
        T = len(self.trees)
        out = np.tile(self.base_scores, (n, 1)).astype(np.float32)   # (n, K)
        if T == 0 or n == 0:
            return out[:, 0] if K == 1 else out
        p = self._pack()
        off, feature, threshold = p["off"], p["feature"], p["threshold"]
        leaf, flags, cat = p["leaf"], p["flags"], p["catMask"]
        # (rows, trees) working set, chunked to bound peak memory.
        chunk = max(1, 2_000_000 // T)
        for s in range(0, n, chunk):
            Xc = X[s:s + chunk]
            m = Xc.shape[0]
            pos = np.broadcast_to(off, (m, T)).copy()   # global node ids
            for _ in range(p["depth"]):
                f = feature[pos]
                split = f >= 0
                if not split.any():
                    break
                v = np.take_along_axis(Xc, np.where(split, f, 0), axis=1)
                fl = flags[pos]
                left = v <= threshold[pos]
                nan = np.isnan(v)
                if nan.any():
                    left = np.where(nan, (fl & 1) != 0, left)
                if cat is not None:
                    cat_nodes = (fl & 2) != 0
                    if cat_nodes.any():
                        data_bins = self.num_bins - 1
                        c = np.where(np.isfinite(v), np.rint(v), -1) \
                              .astype(np.int64)
                        bad = (c < 0) | (c >= data_bins)
                        c = np.where(bad, data_bins, c)
                        words = cat[pos, c >> 5]
                        member = ((words >> (c & 31).astype(np.uint64)) & 1) \
                            .astype(bool)
                        left = np.where(cat_nodes, member, left)
                # children live at 2i+1 / 2i+2 in each tree's local heap
                local = pos - off
                pos = np.where(split, off + 2 * local + 1 + ~left, pos)
            contrib = leaf[pos]                          # (m, trees)
            oc = out[s:s + chunk]
            if K == 1:
                oc[:, 0] += contrib.sum(axis=1, dtype=np.float64) \
                    .astype(np.float32)
            else:
                for k in range(K):     # tree t scores class t % K
                    oc[:, k] += contrib[:, k::K].sum(axis=1, dtype=np.float64) \
                        .astype(np.float32)
        return out[:, 0] if K == 1 else out

    def predict(self, X):
        out = self.predict_raw(X)
        if self.objective in _LOG_LINK_OBJECTIVES:
            out = np.exp(out)
        return out

    def feature_importance(self, importance_type="gain"):
        imp = np.zeros(self.num_features, dtype=np.float32)
        for tree in self.trees:
            splits = tree["feature"] >= 0
            feats = tree["feature"][splits]
            if importance_type == "split":
                np.add.at(imp, feats, 1)
            else:
                np.add.at(imp, feats, tree["gain"][splits]
                          if tree["gain"].size else 0)
        return imp

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self._raw, fh)
