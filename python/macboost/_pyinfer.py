"""Pure-Python (numpy) inference over MacBoost model.json files.

Training requires the Metal core (Apple silicon), but a trained model is
just trees — this backend scores them anywhere Python runs, so models
trained on a Mac deploy to Linux with nothing but `pip install macboost`.
Semantics match the native predictor exactly: `x <= threshold` goes left,
NaN follows the learned default direction, categorical splits test bitmask
membership (unseen/out-of-range categories route like missing), and
poisson/tweedie outputs are exp-transformed.
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

    @property
    def num_trees(self):
        return len(self.trees)

    def predict_raw(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.num_features:
            raise ValueError(
                f"X must be (n, {self.num_features}), got {X.shape}")
        n = X.shape[0]
        K = self.num_classes
        out = np.tile(self.base_scores, (n, 1)).astype(np.float32)   # (n, K)
        for t, tree in enumerate(self.trees):
            node = np.zeros(n, dtype=np.int64)
            active = tree["feature"][node] >= 0
            while active.any():
                idx = np.nonzero(active)[0]
                cur = node[idx]
                v = X[idx, tree["feature"][cur]]
                left = self._goes_left_at(tree, cur, v)
                node[idx] = 2 * cur + 1 + (~left).astype(np.int64)
                active[idx] = tree["feature"][node[idx]] >= 0
            out[:, t % K] += tree["leaf"][node]
        return out[:, 0] if K == 1 else out

    def _goes_left_at(self, tree, node, v):
        thr = tree["threshold"][node]
        flags = tree["flags"][node]
        left = v <= thr
        nan = np.isnan(v)
        if nan.any():
            left = np.where(nan, (flags & 1) != 0, left)
        cat_nodes = (flags & 2) != 0
        if tree["catMask"] is not None and cat_nodes.any():
            data_bins = self.num_bins - 1
            cat = np.where(np.isfinite(v), np.rint(v), -1).astype(np.int64)
            bad = (cat < 0) | (cat >= data_bins)
            cat = np.where(bad, data_bins, cat)
            words = tree["catMask"][node, cat >> 5]
            member = ((words >> (cat & 31).astype(np.uint64)) & 1).astype(bool)
            left = np.where(cat_nodes, member, left)
        return left

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
