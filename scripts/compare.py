# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "lightgbm", "xgboost", "scikit-learn"]
# ///
"""Train LightGBM and XGBoost on the exact dataset exported by the MacBoost
bench (`swift run -c release bench --export <dir>`), with a matching config,
and report fit time + validation RMSE.

Usage: uv run scripts/compare.py <export_dir>
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

d = Path(sys.argv[1] if len(sys.argv) > 1 else "export")
meta = json.loads((d / "meta.json").read_text())
n_tr, n_va, cols = meta["train_rows"], meta["valid_rows"], meta["cols"]

def load(name, rows):
    x = np.fromfile(d / name, dtype=np.float32)
    return x.reshape(cols, rows).T.copy() if x.size == rows * cols else x

Xtr, ytr = load("X_train.bin", n_tr), load("y_train.bin", n_tr)
Xva, yva = load("X_valid.bin", n_va), load("y_valid.bin", n_va)
print(f"loaded: train {Xtr.shape}, valid {Xva.shape}")

trees, depth, lr = meta["trees"], meta["depth"], meta["lr"]
cats = meta.get("categorical", [])
eval_valid = meta.get("eval_valid", False)
if cats:
    print(f"categorical columns: {len(cats)}; per-iteration valid eval: {eval_valid}")

def report(name, fit, predict):
    t0 = time.perf_counter()
    model = fit()
    fit_s = time.perf_counter() - t0
    r = float(np.sqrt(np.mean((predict(model) - yva) ** 2)))
    print(f"{name:>10}: fit {fit_s:6.2f}s   valid RMSE {r:.5f}")

import lightgbm as lgb

def fit_lgb():
    m = lgb.LGBMRegressor(
        n_estimators=trees, max_depth=depth, num_leaves=2**depth,
        learning_rate=lr, max_bin=255, reg_lambda=1.0,
        min_child_samples=1, min_child_weight=1.0, verbose=-1,
    )
    kw = {}
    if cats:
        kw["categorical_feature"] = cats
    if eval_valid:
        kw["eval_set"] = [(Xva, yva)]
        kw["eval_metric"] = "rmse"
    return m.fit(Xtr, ytr, **kw)

report("lightgbm", fit_lgb, lambda m: m.predict(Xva))

import xgboost as xgb

def fit_xgb():
    kw = {}
    if cats:
        kw["enable_categorical"] = True
        kw["feature_types"] = ["c" if i in set(cats) else "q" for i in range(cols)]
    m = xgb.XGBRegressor(
        n_estimators=trees, max_depth=depth, learning_rate=lr,
        tree_method="hist", max_bin=256, reg_lambda=1.0,
        min_child_weight=1.0, verbosity=0,
        eval_metric="rmse", **kw,
    )
    fit_kw = {"verbose": False}
    if eval_valid:
        fit_kw["eval_set"] = [(Xva, yva)]
    return m.fit(Xtr, ytr, **fit_kw)

try:
    report("xgboost", fit_xgb, lambda m: m.predict(Xva))
except Exception as e:
    print(f"   xgboost: skipped ({e})")
