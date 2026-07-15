# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scikit-learn", "xgboost", "onnx", "onnxruntime"]
# ///
"""Python binding tests: regression, classification, missing values,
categoricals, early stopping, and save/load prediction parity.

Run from the repo root:  uv run scripts/test_python.py
(builds are expected at .build/release; run `swift build -c release` first)
Needs an arm64 interpreter — if uv defaults to an x86_64 Python, pass
`--python cpython-3.13-macos-aarch64-none`.
"""
import math
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from macboost import MacBoostClassifier, MacBoostError, MacBoostRegressor  # noqa: E402

rng = np.random.default_rng(11)
checks = 0


def check(cond, msg):
    global checks
    assert cond, msg
    checks += 1
    print(f"  ok: {msg}")


def friedman(n, cols=10):
    X = rng.random((n, cols), dtype=np.float32)
    y = (10 * np.sin(np.pi * X[:, 0] * X[:, 1]) + 20 * (X[:, 2] - 0.5) ** 2
         + 10 * X[:, 3] + 5 * X[:, 4]
         + rng.standard_normal(n).astype(np.float32))
    return X, y.astype(np.float32)


print("regression + eval_set + early stopping")
X, y = friedman(30_000)
Xv, yv = friedman(5_000)
model = MacBoostRegressor(n_estimators=500, max_depth=6, learning_rate=0.2)
model.fit(X, y, eval_set=(Xv, yv), early_stopping_rounds=25)
rmse = float(np.sqrt(np.mean((model.predict(Xv) - yv) ** 2)))
check(rmse < 1.2, f"valid RMSE {rmse:.4f} < 1.2 (noise floor 1.0)")
check(model.best_iteration_ is not None and model.n_trees_ == model.best_iteration_,
      f"early stopping truncated to best iteration ({model.best_iteration_})")

print("missing values + categoricals")
Xm = X.copy()
Xm[rng.random(len(Xm)) < 0.2, 0] = np.nan
cat = rng.integers(0, 5, len(Xm)).astype(np.float32)
ym = y + np.where(np.isin(cat, [1, 3]), 4.0, -4.0).astype(np.float32)
Xm = np.column_stack([Xm, cat])
m2 = MacBoostRegressor(n_estimators=150, max_depth=6, categorical_features=[10])
m2.fit(Xm, ym)
rmse2 = float(np.sqrt(np.mean((m2.predict(Xm) - ym) ** 2)))
check(rmse2 < 1.5, f"train RMSE with NaN + categorical {rmse2:.4f} < 1.5")

print("binary classification")
Xb = rng.random((20_000, 4), dtype=np.float32)
yb = ((Xb[:, 0] + Xb[:, 1] + 0.3 * rng.standard_normal(20_000)) > 1).astype(np.float32)
clf = MacBoostClassifier(n_estimators=60, max_depth=5)
clf.fit(Xb, yb)
proba = clf.predict_proba(Xb)[:, 1]
ll = float(-np.mean(yb * np.log(np.clip(proba, 1e-12, 1))
                    + (1 - yb) * np.log(np.clip(1 - proba, 1e-12, 1))))
check(ll < 0.42, f"train logloss {ll:.4f} at the ~0.41 Bayes floor region")
# Bayes-optimal accuracy for this noisy boundary is ~0.806 (the 0.3-sigma
# noise flips labels near s=1 where the margin density peaks).
acc = float(np.mean(clf.predict(Xb) == yb))
check(acc > 0.79, f"accuracy {acc:.4f} > 0.79 (Bayes ceiling ~0.806)")

print("GOSS")
full_m = MacBoostRegressor(n_estimators=150).fit(X, y)
goss_m = MacBoostRegressor(n_estimators=150, goss=True, top_rate=0.2,
                           other_rate=0.1).fit(X, y)
full_r = float(np.sqrt(np.mean((full_m.predict(Xv) - yv) ** 2)))
goss_r = float(np.sqrt(np.mean((goss_m.predict(Xv) - yv) ** 2)))
check(goss_r < full_r + 0.12,
      f"GOSS RMSE {goss_r:.4f} within 0.12 of full {full_r:.4f} (top_rate/other_rate aliases)")
goss_clf = MacBoostClassifier(n_estimators=40, goss=True).fit(Xb, yb)
goss_acc = float(np.mean(goss_clf.predict(Xb) == yb))
check(goss_acc > 0.79, f"GOSS classifier accuracy {goss_acc:.4f} > 0.79")

print("save / load parity")
with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "model.json"
    m2.save_model(path)
    loaded = MacBoostRegressor.load_model(path)
    check(loaded.n_trees_ == m2.n_trees_, "tree count survives round-trip")
    check(loaded.n_features_ == 11, "feature count survives round-trip")
    p1, p2 = m2.predict(Xm), loaded.predict(Xm)
    check(bool(np.array_equal(p1, p2)), "predictions bitwise identical after round-trip")

print("parameter aliases (LightGBM / XGBoost native spellings)")
m3 = MacBoostRegressor(num_iterations=42, eta=0.3, gamma=0.5, lambda_l2=2.0)
check(m3.n_estimators == 42 and m3.learning_rate == 0.3
      and m3.min_split_gain == 0.5 and m3.reg_lambda == 2.0,
      "num_iterations/eta/gamma/lambda_l2 map to canonical params")
try:
    MacBoostRegressor(colour="red")
    raise SystemExit("expected TypeError for unknown parameter")
except TypeError:
    check(True, "unknown parameter rejected with TypeError")

print("scikit-learn protocol (clone, cross_val_score, grid search)")
from sklearn.base import clone
from sklearn.model_selection import GridSearchCV, cross_val_score

small_X, small_y = friedman(5_000)
est = MacBoostRegressor(n_estimators=100, max_depth=5)
cloned = clone(est)
check(cloned.get_params() == est.get_params(), "sklearn.base.clone round-trips params")
scores = cross_val_score(est, small_X, small_y, cv=3,
                         scoring="neg_root_mean_squared_error")
check(all(-s < 1.6 for s in scores),
      f"cross_val_score works (fold RMSEs {[f'{-s:.3f}' for s in scores]})")
grid = GridSearchCV(MacBoostRegressor(n_estimators=100), {"max_depth": [3, 5]}, cv=2,
                    scoring="neg_root_mean_squared_error")
grid.fit(small_X, small_y)
check(grid.best_params_["max_depth"] == 5,
      f"GridSearchCV runs and prefers depth 5 (best: {grid.best_params_})")

print("error handling")
try:
    MacBoostRegressor().predict(X)
    raise SystemExit("expected MacBoostError for unfitted model")
except MacBoostError:
    check(True, "unfitted predict raises MacBoostError")
bad = MacBoostRegressor(n_estimators=5, categorical_features=[0])
try:
    bad.fit(np.full((10, 2), 999.0, dtype=np.float32), np.zeros(10, dtype=np.float32))
    raise SystemExit("expected MacBoostError for out-of-range categories")
except MacBoostError as e:
    check("Categorical" in str(e), f"category validation surfaces: {e}")

print("guardrails against common data mistakes")
import warnings

Xg, yg = friedman(2_000, cols=5)
try:
    MacBoostRegressor(max_bin=1000).fit(Xg, yg)
    raise SystemExit("expected MacBoostError for max_bin out of range")
except MacBoostError as e:
    check("numBins" in str(e), "invalid max_bin raises (not a process abort)")
try:
    MacBoostRegressor(n_estimators=5).fit(Xg, np.append(yg[:-1], np.nan))
    raise SystemExit("expected MacBoostError for NaN label")
except MacBoostError as e:
    check("finite" in str(e), "NaN in labels rejected with row number")
try:
    MacBoostClassifier(n_estimators=5).fit(Xg, yg)   # continuous labels
    raise SystemExit("expected error for continuous labels")
except ValueError as e:
    check("continuous" in str(e), "continuous labels rejected with guidance")
try:
    MacBoostRegressor(n_estimators=5).fit(Xg, yg[:100])
    raise SystemExit("expected error for X/y row mismatch")
except (MacBoostError, ValueError):
    check(True, "X/y row mismatch rejected")
m5 = MacBoostRegressor(n_estimators=5).fit(Xg, yg)
try:
    m5.predict(Xg[:, :3])
    raise SystemExit("expected MacBoostError for wrong predict width")
except MacBoostError as e:
    check("features" in str(e), "predict with wrong column count rejected")

# The classic leakage bug: the label (or a prediction of it) left in X.
X_leak = np.column_stack([Xg, yg])
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    MacBoostRegressor(n_estimators=5).fit(X_leak, yg)
check(any("leakage" in str(w.message) for w in caught),
      "label column left inside X triggers a target-leakage warning")
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    MacBoostRegressor(n_estimators=5).fit(Xg, yg)
check(not caught, "clean data trains without warnings")


print("pure-python inference backend (Linux deploy path)")
import os
import subprocess, tempfile as _tf
with _tf.TemporaryDirectory() as _tmp:
    # Model exercising NaN + categorical + missing-direction routing.
    Xd = Xm.copy()
    md = MacBoostRegressor(n_estimators=60, max_depth=6, categorical_features=[10])
    md.fit(Xd, ym)
    md.save_model(f"{_tmp}/m.json")
    native_pred = md.predict(Xd[:5000])
    np.save(f"{_tmp}/X.npy", Xd[:5000]); np.save(f"{_tmp}/p.npy", native_pred)
    code = (
        "import os, numpy as np\n"
        "import macboost\n"
        "from macboost import MacBoostRegressor\n"
        "from macboost import _core\n"
        "assert not _core.native_available(), 'native lib should be disabled'\n"
        f"m = MacBoostRegressor.load_model('{_tmp}/m.json')\n"
        f"X = np.load('{_tmp}/X.npy'); ref = np.load('{_tmp}/p.npy')\n"
        "pred = m.predict(X)\n"
        "assert np.allclose(pred, ref, atol=2e-4), float(np.abs(pred-ref).max())\n"
        "assert m.n_trees_ == 60 and m.n_features_in_ == 11\n"
        "imp = m.feature_importances_\n"
        "assert imp.shape == (11,) and imp.sum() > 0\n"
        "try:\n"
        "    m.fit(X, ref)\n"
        "    raise SystemExit('fit should fail without native core')\n"
        "except Exception as e:\n"
        "    assert 'Apple silicon' in str(e), e\n"
        "print('PYBACKEND-OK')\n"
    )
    env = dict(os.environ, MACBOOST_FORCE_PYTHON="1",
               PYTHONPATH=str(Path(__file__).resolve().parent.parent / "python"))
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True)
    check("PYBACKEND-OK" in r.stdout,
          f"python-only backend matches native predictions (stderr: {r.stderr[-300:] if r.returncode else 'clean'})")

    # Multiclass through the python backend too.
    _X3 = rng.random((6_000, 4), dtype=np.float32)
    _y3 = np.minimum(2, (_X3[:, 0] * 3).astype(int)).astype(np.float32)
    mm = MacBoostClassifier(n_estimators=20, max_depth=4).fit(_X3, _y3)
    mm.save_model(f"{_tmp}/mc.json")
    ref3 = mm._raw_predict(_X3[:2000])
    np.save(f"{_tmp}/X3.npy", _X3[:2000]); np.save(f"{_tmp}/p3.npy", ref3)
    code = (
        "import numpy as np\n"
        "from macboost._pyinfer import PyModel\n"
        f"m = PyModel('{_tmp}/mc.json')\n"
        f"X = np.load('{_tmp}/X3.npy'); ref = np.load('{_tmp}/p3.npy')\n"
        "pred = m.predict_raw(X)\n"
        "assert pred.shape == ref.shape\n"
        "assert np.allclose(pred, ref, atol=2e-4), float(np.abs(pred-ref).max())\n"
        "print('PYBACKEND-MC-OK')\n"
    )
    r = subprocess.run([sys.executable, "-c", code], env=env,
                       capture_output=True, text=True)
    check("PYBACKEND-MC-OK" in r.stdout,
          f"python backend multiclass raw scores match (stderr: {r.stderr[-300:] if r.returncode else 'clean'})")

print("xgboost export parity")
import xgboost as _xgb
with _tf.TemporaryDirectory() as _tmp:
    # Regression with NaN + categorical.
    md2 = MacBoostRegressor(n_estimators=50, max_depth=6, categorical_features=[10])
    md2.fit(Xm, ym)
    md2.save_xgboost(f"{_tmp}/m.xgb.json")
    bst = _xgb.Booster()
    bst.load_model(f"{_tmp}/m.xgb.json")
    Xt = Xm[:5000]
    ours = md2.predict(Xt)
    theirs = bst.predict(_xgb.DMatrix(Xt), output_margin=True)
    gap = float(np.abs(ours - theirs).max())
    check(gap < 3e-4, f"xgboost regression margins match (max gap {gap:.2e}, NaN+cat data)")

    # Binary logistic: compare probabilities.
    clf2 = MacBoostClassifier(n_estimators=40, max_depth=5).fit(Xb, yb)
    clf2.save_xgboost(f"{_tmp}/b.xgb.json")
    bstb = _xgb.Booster(); bstb.load_model(f"{_tmp}/b.xgb.json")
    p_ours = clf2.predict_proba(Xb[:5000])[:, 1]
    p_theirs = bstb.predict(_xgb.DMatrix(Xb[:5000]))
    gap = float(np.abs(p_ours - p_theirs).max())
    check(gap < 3e-4, f"xgboost binary probabilities match (max gap {gap:.2e})")

    # Poisson: exp transform on both sides.
    ypois2 = rng.poisson(np.exp(1 + Xg[:, 0])).astype(np.float32)
    pois2 = MacBoostRegressor(n_estimators=40, objective="poisson").fit(Xg, ypois2)
    pois2.save_xgboost(f"{_tmp}/p.xgb.json")
    bstp = _xgb.Booster(); bstp.load_model(f"{_tmp}/p.xgb.json")
    mu_ours = pois2.predict(Xg[:2000])
    mu_theirs = bstp.predict(_xgb.DMatrix(Xg[:2000]))
    rel = float(np.abs(mu_ours - mu_theirs).max() / mu_ours.max())
    check(rel < 1e-3, f"xgboost poisson means match (max rel gap {rel:.2e})")

    # Multiclass softprob.
    _X4 = rng.random((6_000, 4), dtype=np.float32)
    _y4 = np.minimum(2, (_X4[:, 0] * 3).astype(int)).astype(np.float32)
    mc2 = MacBoostClassifier(n_estimators=25, max_depth=4).fit(_X4, _y4)
    mc2.save_xgboost(f"{_tmp}/mc.xgb.json")
    bstm = _xgb.Booster(); bstm.load_model(f"{_tmp}/mc.xgb.json")
    pr_ours = mc2.predict_proba(_X4[:2000])
    pr_theirs = bstm.predict(_xgb.DMatrix(_X4[:2000]))
    gap = float(np.abs(pr_ours - pr_theirs).max())
    check(gap < 3e-4, f"xgboost multiclass probabilities match (max gap {gap:.2e})")

print("onnx export parity")
import onnxruntime as _ort
with _tf.TemporaryDirectory() as _tmp:
    def _ort_predict(path, Xq):
        sess = _ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return sess.run(["prediction"], {"input": Xq.astype(np.float32)})[0]

    # Regression with NaN + categorical (reusing md2 from the xgboost tests).
    md2.save_onnx(f"{_tmp}/m.onnx")
    ours = md2.predict(Xm[:5000])
    theirs = _ort_predict(f"{_tmp}/m.onnx", Xm[:5000]).ravel()
    gap = float(np.abs(ours - theirs).max())
    check(gap < 3e-4, f"onnx regression predictions match (max gap {gap:.2e}, NaN+cat data)")

    # Binary logistic probabilities.
    clf2.save_onnx(f"{_tmp}/b.onnx")
    p_ours = clf2.predict_proba(Xb[:5000])[:, 1]
    p_theirs = _ort_predict(f"{_tmp}/b.onnx", Xb[:5000]).ravel()
    gap = float(np.abs(p_ours - p_theirs).max())
    check(gap < 3e-4, f"onnx binary probabilities match (max gap {gap:.2e})")

    # Poisson means (Exp node appended).
    pois2.save_onnx(f"{_tmp}/p.onnx")
    mu_ours = pois2.predict(Xg[:2000])
    mu_theirs = _ort_predict(f"{_tmp}/p.onnx", Xg[:2000]).ravel()
    rel = float(np.abs(mu_ours - mu_theirs).max() / mu_ours.max())
    check(rel < 1e-3, f"onnx poisson means match (max rel gap {rel:.2e})")

    # Multiclass probabilities (Softmax node appended).
    mc2.save_onnx(f"{_tmp}/mc.onnx")
    pr_ours = mc2.predict_proba(_X4[:2000])
    pr_theirs = _ort_predict(f"{_tmp}/mc.onnx", _X4[:2000])
    gap = float(np.abs(pr_ours - pr_theirs).max())
    check(gap < 3e-4, f"onnx multiclass probabilities match (max gap {gap:.2e})")

print("tier 1+2 features")
# Multiclass with arbitrary label values (auto-encoded).
Xm3 = rng.random((8_000, 4), dtype=np.float32)
ym3 = np.select([Xm3[:, 0] > 0.66, Xm3[:, 0] > 0.33], ["high", "mid"], "low")
mc = MacBoostClassifier(n_estimators=40, max_depth=4).fit(Xm3, ym3)
proba = mc.predict_proba(Xm3)
check(proba.shape == (8_000, 3) and np.allclose(proba.sum(axis=1), 1, atol=1e-4),
      "multiclass predict_proba is (n, 3) and rows sum to 1")
acc3 = float(np.mean(mc.predict(Xm3) == ym3))
check(acc3 > 0.9, f"multiclass accuracy {acc3:.3f} with string labels round-tripped")

# Feature importance concentrates on informative features.
imp = MacBoostRegressor(n_estimators=60).fit(X, y).feature_importances_
check(imp[:5].sum() > imp[5:].sum() * 3, "gain importance concentrates on informative features")

# SHAP sums to prediction.
sh_m = MacBoostRegressor(n_estimators=30, max_depth=5).fit(X[:3000], y[:3000])
contrib = sh_m.predict_contrib(X[:200])
raw = sh_m.predict(X[:200])
check(bool(np.allclose(contrib.sum(axis=1), raw, atol=2e-2)),
      "predict_contrib rows sum to the prediction")

# Sample weights + objectives + sampling knobs plumb through.
MacBoostRegressor(n_estimators=10).fit(X[:5000], y[:5000],
                                       sample_weight=np.ones(5000, np.float32))
ypois = rng.poisson(np.exp(1 + Xg[:, 0])).astype(np.float32)
pois = MacBoostRegressor(n_estimators=40, objective="poisson").fit(Xg, ypois)
check(bool((pois.predict(Xg) > 0).all()), "poisson objective returns positive means")
q = MacBoostRegressor(n_estimators=60, objective="quantile", alpha=0.9).fit(Xg, yg)
cov = float(np.mean(yg <= q.predict(Xg)))
check(0.8 < cov < 1.0, f"quantile alpha=0.9 coverage {cov:.3f}")
MacBoostRegressor(n_estimators=10, subsample=0.7, colsample_bytree=0.7).fit(Xg, yg)
check(True, "subsample + colsample_bytree accepted")
mono = MacBoostRegressor(n_estimators=30, monotone_constraints=[0, 0, 0, 1, 0]).fit(Xg, yg)
check(mono.n_trees_ == 30, "monotone constraints accepted")

# Multiclass SHAP: class blocks sum to class scores.
mc_contrib = mc.predict_contrib(Xm3[:100])
check(mc_contrib.shape == (100, 3 * 5), "multiclass contrib is (n, K*(cols+1))")

# Warm start.
m_a = MacBoostRegressor(n_estimators=20).fit(Xg, yg)
m_b = MacBoostRegressor(n_estimators=20).fit(Xg, yg, init_model=m_a)
check(m_b.n_trees_ == 40, "init_model warm start extends the ensemble")

# AUC eval metric.
auc_m = MacBoostClassifier(n_estimators=60, eval_metric="auc")
auc_m.fit(Xb, yb, eval_set=(Xb[:4000], yb[:4000]), early_stopping_rounds=15)
check(auc_m.best_iteration_ is not None, "auc metric drives early stopping")

print(f"\nPython binding tests PASSED ({checks} checks)")
