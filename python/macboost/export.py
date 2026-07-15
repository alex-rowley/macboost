"""Export MacBoost models for deployment on non-Apple platforms.

- save_xgboost(model, path): writes an XGBoost JSON model that standard
  `xgboost` loads and scores anywhere (Linux servers, containers, ...).
- to_onnx(model): builds an ONNX graph using the ai.onnx.ml TreeEnsemble
  operators for onnxruntime everywhere (requires the `onnx` package).

Semantics notes:
- MacBoost splits are `x <= t` (left); XGBoost/ONNX-LEQ conventions are
  matched exactly (thresholds nudged by one float32 ulp where needed).
- Categorical subset splits have no direct equivalent in either target, so
  they are rewritten into chains of numeric interval tests routing to
  duplicated subtrees — identical predictions, larger trees. Unseen or
  out-of-range categories route like missing values, as in MacBoost.
- Raw-score regression objectives (l2/mae/huber/quantile) export as plain
  regression; logistic/poisson/tweedie/multiclass map to the target's
  equivalent transform. Base scores are folded into the first tree's
  leaves so no cross-library base-score semantics are involved.
"""

import json
import tempfile

import numpy as np

_FLAG_DEFAULT_LEFT = 1
_FLAG_CATEGORICAL = 2


def _model_dict(model):
    """Model JSON dict from an estimator or a PyModel."""
    pymodel = getattr(model, "_pymodel", None)
    if pymodel is not None:
        return pymodel._raw
    with tempfile.NamedTemporaryFile(suffix=".json") as fh:
        model.save_model(fh.name)
        with open(fh.name) as f:
            return json.load(f)


class _Node:
    __slots__ = ("feature", "threshold", "default_left", "left", "right", "value")

    def __init__(self, feature=-1, threshold=0.0, default_left=False,
                 left=-1, right=-1, value=0.0):
        self.feature = feature          # -1 => leaf
        self.threshold = threshold      # "x <= threshold" goes left
        self.default_left = default_left
        self.left = left
        self.right = right
        self.value = value


def _explicit_tree(tree, num_bins):
    """Heap-layout tree -> explicit numeric-only node list (categorical
    splits expanded into interval chains with duplicated subtrees)."""
    feature = tree["feature"]
    threshold = tree["threshold"]
    leaf = tree["leaf"]
    flags = tree["flags"]
    cat_mask = tree.get("catMask") or []
    data_bins = num_bins - 1
    nodes = []

    def add(node):
        nodes.append(node)
        return len(nodes) - 1

    def member(heap_idx, cat):
        word = cat_mask[heap_idx * 8 + (cat >> 5)]
        return (word >> (cat & 31)) & 1 == 1

    def build(h):
        f = int(feature[h])
        if f < 0:
            return add(_Node(value=float(leaf[h])))
        fl = int(flags[h])
        if fl & _FLAG_CATEGORICAL == 0:
            n = add(_Node(feature=f, threshold=float(threshold[h]),
                          default_left=bool(fl & _FLAG_DEFAULT_LEFT)))
            nodes[n].left = build(2 * h + 1)
            nodes[n].right = build(2 * h + 2)
            return n

        # Categorical: membership of ints 0..data_bins-1 (missing bin =
        # data_bins participates too). Compress into runs of equal side,
        # then a right-deep chain of `x <= run_end + 0.5` tests. NaN and
        # out-of-range categories route to the missing bin's side.
        sides = [member(h, c) for c in range(data_bins)]     # True => left
        missing_left = member(h, data_bins)
        runs = []                                            # (end_cat, side)
        for c in range(data_bins):
            if runs and runs[-1][1] == sides[c]:
                runs[-1] = (c, sides[c])
            else:
                runs.append((c, sides[c]))

        def side_subtree(left_side):
            return build(2 * h + 1) if left_side else build(2 * h + 2)

        # Trailing destination for x > last category: missing side.
        chain = side_subtree(missing_left)
        for end, side in reversed(runs):
            n = add(_Node(feature=f, threshold=end + 0.5,
                          default_left=(side == missing_left)))
            nodes[n].left = side_subtree(side)
            nodes[n].right = chain
            chain = n
        # Leading guard: x < 0 (invalid category) -> missing side.
        n = add(_Node(feature=f, threshold=-0.5, default_left=missing_left))
        nodes[n].left = side_subtree(missing_left)
        nodes[n].right = chain
        return n

    root = build(0)
    # BFS relabel: XGBoost allocates parents before children and its JSON
    # loader relies on that ordering, so match it (root at index 0).
    order = [root]
    for i in order:
        if nodes[i].feature >= 0:
            order.append(nodes[i].left)
            order.append(nodes[i].right)
    remap = {old: new for new, old in enumerate(order)}
    reordered = [nodes[old] for old in order]
    for node in reordered:
        if node.feature >= 0:
            node.left = remap[node.left]
            node.right = remap[node.right]
    return reordered


_NEUTRAL_BASE = {          # base_score whose margin is 0, per objective
    "reg:squarederror": "0", "binary:logistic": "0.5",
    "count:poisson": "1", "reg:tweedie": "1", "multi:softprob": "0.5",
}


def _xgb_objective(m):
    obj = int(m["objective"])
    if obj == 1:
        return "binary:logistic", {"reg_loss_param": {"scale_pos_weight": "1"}}
    if obj == 5:
        return "count:poisson", {"poisson_regression_param": {"max_delta_step": "0.7"}}
    if obj == 6:
        return "reg:tweedie", {"tweedie_regression_param": {"tweedie_variance_power": "1.5"}}
    if obj == 7:
        k = str(int(m.get("numClasses") or 1))
        return "multi:softprob", {"softmax_multiclass_param": {"num_class": k}}
    # l2 / mae / huber / quantile: raw scores, plain regression at inference.
    return "reg:squarederror", {"reg_loss_param": {"scale_pos_weight": "1"}}


def to_xgboost_dict(model):
    m = _model_dict(model)
    num_features = int(m["numFeatures"])
    num_bins = int(m["numBins"])
    k = int(m.get("numClasses") or 1)
    base_scores = m.get("baseScores") or [m["baseScore"]]
    obj_name, obj_param = _xgb_objective(m)

    trees_json = []
    for t, tree in enumerate(m["trees"]):
        nodes = _explicit_tree(tree, num_bins)
        n = len(nodes)
        left = [node.left for node in nodes]
        right = [node.right for node in nodes]
        parents = [2147483647] * n
        for i, node in enumerate(nodes):
            if node.feature >= 0:
                parents[node.left] = i
                parents[node.right] = i
        conditions, indices, defaults, weights = [], [], [], []
        for node in nodes:
            if node.feature >= 0:
                # xgb: x < condition goes left; ours: x <= t. Nudge up 1 ulp.
                thr = float(np.nextafter(np.float32(node.threshold),
                                         np.float32(np.inf)))
                conditions.append(thr)
                indices.append(node.feature)
                defaults.append(1 if node.default_left else 0)
                weights.append(0.0)
            else:
                value = node.value
                # Fold per-class base scores into the first tree of each class.
                if t < k:
                    value += float(base_scores[t % k])
                conditions.append(value)
                indices.append(0)
                defaults.append(0)
                weights.append(value)
        trees_json.append({
            "base_weights": weights,
            "categories": [], "categories_nodes": [],
            "categories_segments": [], "categories_sizes": [],
            "default_left": defaults,
            "id": t,
            "left_children": left,
            "loss_changes": [0.0] * n,
            "parents": parents,
            "right_children": right,
            "split_conditions": conditions,
            "split_indices": indices,
            "split_type": [0] * n,
            "sum_hessian": [1.0] * n,
            "tree_param": {
                "num_deleted": "0",
                "num_feature": str(num_features),
                "num_nodes": str(n),
                "size_leaf_vector": "1",
            },
        })

    num_trees = len(trees_json)
    rounds = num_trees // k if k else num_trees
    return {
        "learner": {
            "attributes": {},
            "feature_names": [],
            "feature_types": [],
            "gradient_booster": {
                "model": {
                    "gbtree_model_param": {
                        "num_parallel_tree": "1",
                        "num_trees": str(num_trees),
                    },
                    "iteration_indptr": [i * k for i in range(rounds + 1)],
                    "tree_info": [t % k for t in range(num_trees)],
                    "trees": trees_json,
                },
                "name": "gbtree",
            },
            "learner_model_param": {
                "base_score": _NEUTRAL_BASE[obj_name],
                "boost_from_average": "1",
                "num_class": str(k if k > 1 else 0),
                "num_feature": str(num_features),
                "num_target": "1",
            },
            "objective": {"name": obj_name, **obj_param},
        },
        "version": [2, 1, 0],
    }


def save_xgboost(model, path):
    """Write an XGBoost-format JSON model loadable by xgb.Booster()."""
    with open(path, "w") as fh:
        json.dump(to_xgboost_dict(model), fh)


def to_onnx(model):
    """Build an ONNX ModelProto (ai.onnx.ml TreeEnsembleRegressor).

    The graph takes float input `(n, num_features)` and produces
    `prediction`: raw scores for regression objectives, probabilities for
    logistic (n, 1) / multiclass (n, K), and exp-transformed means for
    poisson/tweedie. Requires the `onnx` package.
    """
    try:
        from onnx import TensorProto, helper
    except ImportError as e:
        raise ImportError(
            "ONNX export requires the `onnx` package: pip install onnx") from e

    m = _model_dict(model)
    num_features = int(m["numFeatures"])
    num_bins = int(m["numBins"])
    k = int(m.get("numClasses") or 1)
    obj = int(m["objective"])
    base_scores = [float(b) for b in (m.get("baseScores") or [m["baseScore"]])]

    attrs = {
        "nodes_treeids": [], "nodes_nodeids": [], "nodes_featureids": [],
        "nodes_modes": [], "nodes_values": [], "nodes_truenodeids": [],
        "nodes_falsenodeids": [], "nodes_missing_value_tracks_true": [],
        "target_treeids": [], "target_nodeids": [], "target_ids": [],
        "target_weights": [],
    }
    for t, tree in enumerate(m["trees"]):
        for i, node in enumerate(_explicit_tree(tree, num_bins)):
            attrs["nodes_treeids"].append(t)
            attrs["nodes_nodeids"].append(i)
            if node.feature >= 0:
                attrs["nodes_featureids"].append(node.feature)
                attrs["nodes_modes"].append("BRANCH_LEQ")   # v <= t: exact match
                attrs["nodes_values"].append(node.threshold)
                attrs["nodes_truenodeids"].append(node.left)
                attrs["nodes_falsenodeids"].append(node.right)
                attrs["nodes_missing_value_tracks_true"].append(
                    1 if node.default_left else 0)
            else:
                attrs["nodes_featureids"].append(0)
                attrs["nodes_modes"].append("LEAF")
                attrs["nodes_values"].append(0.0)
                attrs["nodes_truenodeids"].append(0)
                attrs["nodes_falsenodeids"].append(0)
                attrs["nodes_missing_value_tracks_true"].append(0)
                attrs["target_treeids"].append(t)
                attrs["target_nodeids"].append(i)
                attrs["target_ids"].append(t % k)
                attrs["target_weights"].append(node.value)

    ensemble = helper.make_node(
        "TreeEnsembleRegressor", ["input"], ["margin"],
        domain="ai.onnx.ml", n_targets=k, base_values=base_scores,
        post_transform="NONE", **attrs)
    nodes = [ensemble]
    if obj == 1:                                    # logistic -> probability
        nodes.append(helper.make_node("Sigmoid", ["margin"], ["prediction"]))
    elif obj == 7:                                  # multiclass -> probabilities
        nodes.append(helper.make_node("Softmax", ["margin"], ["prediction"], axis=1))
    elif obj in (5, 6):                             # poisson/tweedie -> mean
        nodes.append(helper.make_node("Exp", ["margin"], ["prediction"]))
    else:
        nodes.append(helper.make_node("Identity", ["margin"], ["prediction"]))

    graph = helper.make_graph(
        nodes, "macboost",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT,
                                       [None, num_features])],
        [helper.make_tensor_value_info("prediction", TensorProto.FLOAT,
                                       [None, k])],
    )
    onnx_model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("ai.onnx.ml", 1),
                              helper.make_opsetid("", 13)],
        producer_name="macboost", ir_version=7)
    return onnx_model


def save_onnx(model, path):
    """Write an ONNX model scoreable by onnxruntime on any platform."""
    with open(path, "wb") as fh:
        fh.write(to_onnx(model).SerializeToString())
