"""macboost — gradient boosted trees on Apple-silicon GPUs (Metal).

Scikit-learn-compatible estimators backed by the Swift/Metal core:

    from macboost import MacBoostRegressor
    model = MacBoostRegressor(n_estimators=200, max_depth=6)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid),
              early_stopping_rounds=50)
    preds = model.predict(X_test)

Parameter names follow the dialect LightGBM's and XGBoost's sklearn wrappers
share; the well-known native spellings (num_iterations, eta, gamma,
lambda_l2, ...) are accepted as aliases. get_params/set_params implement the
scikit-learn estimator protocol, so clone, Pipeline, GridSearchCV and
cross_val_score work.
"""

import numpy as np

from . import _core
from ._core import MacBoostError

# scikit-learn is optional: when present, inheriting its base classes gives
# full estimator-protocol compliance (__sklearn_tags__, clone, CV tooling).
try:
    from sklearn.base import BaseEstimator as _SkBase
    from sklearn.base import ClassifierMixin as _SkClassifier
    from sklearn.base import RegressorMixin as _SkRegressor
except ImportError:
    class _SkBase: pass
    class _SkRegressor: pass
    class _SkClassifier: pass

__all__ = ["MacBoostRegressor", "MacBoostClassifier", "MacBoostError"]
__version__ = "0.2.0"

# Native LightGBM / XGBoost spellings -> canonical macboost names.
_ALIASES = {
    "num_trees": "n_estimators",
    "num_iterations": "n_estimators",
    "num_boost_round": "n_estimators",
    "num_round": "n_estimators",
    "eta": "learning_rate",
    "shrinkage_rate": "learning_rate",
    "lambda_l2": "reg_lambda",
    "reg_lambda_l2": "reg_lambda",
    "gamma": "min_split_gain",
    "min_split_loss": "min_split_gain",
    "min_gain_to_split": "min_split_gain",
    "min_sum_hessian_in_leaf": "min_child_weight",
    "max_bins": "max_bin",
    "categorical_feature": "categorical_features",
    "top_rate": "goss_top_rate",
    "other_rate": "goss_other_rate",
    "bagging_fraction": "subsample",
    "feature_fraction": "colsample_bytree",
    "colsample": "colsample_bytree",
    "eval_metric": "metric",
    "scale_pos_weight": "scale_pos_weight",
    "monotone_constraints": "monotone_constraints",
    "tweedie_variance_power": "tweedie_variance_power",
}

_PARAM_NAMES = (
    "n_estimators", "max_depth", "learning_rate", "reg_lambda",
    "min_child_weight", "min_split_gain", "max_bin", "categorical_features",
    "cat_smooth", "early_stopping_rounds", "verbose",
    "goss", "goss_top_rate", "goss_other_rate",
    "objective", "alpha", "tweedie_variance_power", "scale_pos_weight",
    "subsample", "colsample_bytree", "monotone_constraints", "metric",
    "importance_type",
)


def _canonical(name):
    return _ALIASES.get(name, name)


class _BaseBooster:
    _objective = "regression"

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1,
                 reg_lambda=1.0, min_child_weight=1.0, min_split_gain=0.0,
                 max_bin=256, categorical_features=None, cat_smooth=10.0,
                 early_stopping_rounds=0, verbose=False,
                 goss=False, goss_top_rate=0.2, goss_other_rate=0.1,
                 objective=None, alpha=0.9, tweedie_variance_power=1.5,
                 scale_pos_weight=1.0, subsample=1.0, colsample_bytree=1.0,
                 monotone_constraints=None, metric="auto",
                 importance_type="gain", **aliases):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.reg_lambda = reg_lambda
        self.min_child_weight = min_child_weight
        self.min_split_gain = min_split_gain
        self.max_bin = max_bin
        self.categorical_features = categorical_features
        self.cat_smooth = cat_smooth
        self.early_stopping_rounds = early_stopping_rounds
        self.verbose = verbose
        self.goss = goss
        self.goss_top_rate = goss_top_rate
        self.goss_other_rate = goss_other_rate
        self.objective = objective
        self.alpha = alpha
        self.tweedie_variance_power = tweedie_variance_power
        self.scale_pos_weight = scale_pos_weight
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.monotone_constraints = monotone_constraints
        self.metric = metric
        self.importance_type = importance_type
        self._handle = None
        self.classes_ = None
        for key, value in aliases.items():
            canon = _canonical(key)
            if canon not in _PARAM_NAMES:
                raise TypeError(f"unknown parameter '{key}'")
            setattr(self, canon, value)

    # -- scikit-learn estimator protocol ---------------------------------
    def get_params(self, deep=True):
        return {k: getattr(self, k) for k in _PARAM_NAMES}

    def set_params(self, **params):
        for key, value in params.items():
            canon = _canonical(key)
            if canon not in _PARAM_NAMES:
                raise ValueError(f"unknown parameter '{key}' for {type(self).__name__}")
            setattr(self, canon, value)
        return self

    # ---------------------------------------------------------------------
    def _config(self, early_stopping_rounds, eval_every):
        objective = self.objective or self._objective
        cfg = {
            "num_trees": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "lambda": self.reg_lambda,
            "min_child_weight": self.min_child_weight,
            "min_split_gain": self.min_split_gain,
            "num_bins": self.max_bin,
            "objective": objective,
            "cat_smooth": self.cat_smooth,
            "early_stopping_rounds": early_stopping_rounds,
            "eval_every": eval_every,
            "verbose": self.verbose,
            "goss": self.goss,
            "goss_top_rate": self.goss_top_rate,
            "goss_other_rate": self.goss_other_rate,
            "alpha": self.alpha,
            "tweedie_variance_power": self.tweedie_variance_power,
            "scale_pos_weight": self.scale_pos_weight,
            "subsample": self.subsample,
            "feature_fraction": self.colsample_bytree,
        }
        if self.metric and self.metric != "auto":
            cfg["metric"] = self.metric
        if self.monotone_constraints:
            cfg["monotone_constraints"] = list(self.monotone_constraints)
        if self.categorical_features:
            cfg["categorical_features"] = list(self.categorical_features)
        return cfg

    def fit(self, X, y, eval_set=None, early_stopping_rounds=None, eval_every=0,
            sample_weight=None, init_model=None):
        if not _core.native_available():
            raise MacBoostError(
                "training requires the native Metal core (Apple silicon, "
                "macOS 15+). This install supports inference only — use "
                "load_model() to score models trained on a Mac.")
        """Train. X: (n, f) array; NaN marks missing values. Columns listed
        in categorical_features must hold integer category ids.

        eval_set: (X_valid, y_valid) tuple, or [(X_valid, y_valid)] in the
        LightGBM/XGBoost list style. early_stopping_rounds here overrides
        the constructor value (both places accepted, like XGBoost)."""
        if isinstance(eval_set, list):
            if len(eval_set) != 1:
                raise ValueError("only a single eval_set is supported")
            eval_set = eval_set[0]
        rounds = (self.early_stopping_rounds if early_stopping_rounds is None
                  else early_stopping_rounds)
        if rounds and eval_set is None:
            raise ValueError("early stopping requires an eval_set")
        cfg = self._config(rounds, eval_every)
        y = self._encode_labels(y)
        if eval_set is not None and self.classes_ is not None and len(self.classes_) > 0:
            ev = eval_set if not isinstance(eval_set, list) else eval_set[0]
            eval_set = (ev[0], self._transform_labels(ev[1]))
        if self.classes_ is not None and len(self.classes_) > 2:
            cfg["objective"] = "multiclass"
            cfg["num_class"] = len(self.classes_)
        import tempfile as _tmp
        init_path = None
        if init_model is not None:
            if isinstance(init_model, (str, bytes)) or hasattr(init_model, "__fspath__"):
                cfg["init_model"] = str(init_model)
            else:
                fd = _tmp.NamedTemporaryFile(suffix=".json", delete=False)
                init_path = fd.name
                fd.close()
                init_model.save_model(init_path)
                cfg["init_model"] = init_path
        try:
            self._handle = _core.train(cfg, X, y, eval_set, sample_weight)
        finally:
            if init_path:
                import os as _os
                _os.unlink(init_path)
        import warnings as _warnings
        for w in _core.warnings_of(self._handle):
            _warnings.warn(w, UserWarning, stacklevel=2)
        return self

    def _raw_predict(self, X):
        if getattr(self, "_pymodel", None) is not None:
            return self._pymodel.predict_raw(np.asarray(X, dtype=np.float32))
        if self._handle is None:
            raise MacBoostError("model is not fitted; call fit() or load_model()")
        return _core.predict(self._handle, X)

    def _encode_labels(self, y):
        return np.asarray(y, dtype=np.float32).ravel()

    def _transform_labels(self, y):
        return y

    def feature_importance(self, importance_type=None):
        if getattr(self, "_pymodel", None) is not None:
            return self._pymodel.feature_importance(
                importance_type or self.importance_type)
        if self._handle is None:
            raise MacBoostError("model is not fitted")
        return _core.feature_importance(
            self._handle, importance_type or self.importance_type)

    @property
    def feature_importances_(self):
        return self.feature_importance()

    def predict_contrib(self, X):
        """SHAP contributions, rows x (n_features + 1); the last column is
        the expected value. Rows sum to the raw prediction."""
        if getattr(self, "_pymodel", None) is not None:
            raise MacBoostError(
                "predict_contrib requires the native core; export the model "
                "or score contributions on a Mac")
        if self._handle is None:
            raise MacBoostError("model is not fitted")
        return _core.predict_contrib(self._handle, X)

    def save_xgboost(self, path):
        """Export as an XGBoost JSON model for deployment anywhere xgboost
        runs (identical predictions; categorical splits are expanded into
        equivalent numeric chains)."""
        from . import export
        export.save_xgboost(self, path)

    def save_onnx(self, path):
        """Export as an ONNX model (ai.onnx.ml TreeEnsemble) scoreable by
        onnxruntime on any platform. Requires the `onnx` package."""
        from . import export
        export.save_onnx(self, path)

    def save_model(self, path):
        if getattr(self, "_pymodel", None) is not None:
            return self._pymodel.save(path)
        if self._handle is None:
            raise MacBoostError("model is not fitted")
        _core.save(self._handle, path)

    @classmethod
    def load_model(cls, path):
        model = cls()
        if _core.native_available():
            model._handle = _core.load(path)
        else:
            # Pure-Python inference backend: works on any platform.
            from ._pyinfer import PyModel
            model._pymodel = PyModel(path)
        return model

    @property
    def n_trees_(self):
        if getattr(self, "_pymodel", None) is not None:
            return self._pymodel.num_trees
        return _core.num_trees(self._handle) if self._handle else 0

    @property
    def n_features_in_(self):
        if getattr(self, "_pymodel", None) is not None:
            return self._pymodel.num_features
        return _core.num_features(self._handle) if self._handle else 0

    # Back-compat spelling.
    n_features_ = n_features_in_

    @property
    def best_iteration_(self):
        best = _core.best_iteration(self._handle) if self._handle else 0
        return best or None


class MacBoostRegressor(_SkRegressor, _BaseBooster, _SkBase):
    _objective = "regression"
    _estimator_type = "regressor"

    def predict(self, X):
        if getattr(self, "_pymodel", None) is not None:
            return self._pymodel.predict(np.asarray(X, dtype=np.float32))
        return self._raw_predict(X)

    def score(self, X, y):
        """R^2, per the scikit-learn regressor contract."""
        y = np.asarray(y, dtype=np.float64).ravel()
        pred = self.predict(X).astype(np.float64)
        ss_res = float(np.sum((y - pred) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class MacBoostClassifier(_SkClassifier, _BaseBooster, _SkBase):
    _objective = "binary"
    _estimator_type = "classifier"

    def _encode_labels(self, y):
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        if len(self.classes_) < 2:
            raise ValueError("classifier needs at least 2 classes")
        if len(self.classes_) > 128:
            raise ValueError(
                f"y has {len(self.classes_)} distinct values — is it continuous? "
                "Use MacBoostRegressor for regression targets")
        return self._transform_labels(y)

    def _transform_labels(self, y):
        y = np.asarray(y).ravel()
        lookup = {c: i for i, c in enumerate(self.classes_)}
        return np.asarray([lookup[v] for v in y], dtype=np.float32)

    def predict_proba(self, X):
        raw = self._raw_predict(X).astype(np.float64)
        if raw.ndim == 2:                       # multiclass softmax
            e = np.exp(raw - raw.max(axis=1, keepdims=True))
            return e / e.sum(axis=1, keepdims=True)
        p = 1.0 / (1.0 + np.exp(-raw))
        return np.column_stack([1 - p, p])

    def predict(self, X):
        idx = np.argmax(self.predict_proba(X), axis=1)
        classes = self.classes_ if self.classes_ is not None else np.array([0, 1])
        return classes[idx]

    def score(self, X, y):
        """Accuracy, per the scikit-learn classifier contract."""
        return float(np.mean(self.predict(X) == np.asarray(y).ravel()))
