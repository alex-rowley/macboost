# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "lightgbm", "xgboost", "scikit-learn"]
# ///
"""HIGGS benchmark (UCI, 11M rows x 28 features, binary classification) —
the canonical real-world GBDT benchmark from the XGBoost/LightGBM papers.

Downloads to ./higgs-cache on first run (~2.6GB) and caches the parsed
matrices as .npy. Standard protocol: last 500k rows are the test set.

Usage:  swift build -c release && uv run scripts/bench_higgs.py [cache_dir]
"""
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00280/HIGGS.csv.gz"
cache = Path(sys.argv[1] if len(sys.argv) > 1 else "higgs-cache")
cache.mkdir(exist_ok=True)

if not (cache / "X.npy").exists():
    gz = cache / "HIGGS.csv.gz"
    if not gz.exists():
        print(f"downloading {URL} ...")
        urllib.request.urlretrieve(URL, gz)
    print("parsing CSV (one-time, a few minutes) ...")
    import pandas as pd
    df = pd.read_csv(gz, header=None, dtype=np.float32)
    np.save(cache / "y.npy", df.iloc[:, 0].to_numpy(np.float32))
    np.save(cache / "X.npy", df.iloc[:, 1:].to_numpy(np.float32))
    del df

X = np.load(cache / "X.npy")
y = np.load(cache / "y.npy")
Xtr, ytr = X[:-500_000], y[:-500_000]
Xte, yte = X[-500_000:], y[-500_000:]
print(f"HIGGS: train {Xtr.shape}, test {Xte.shape}")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "python"))
from macboost import MacBoostClassifier  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

# Matched config across libraries: 100 trees, depth 6, lr 0.1, 255/256 bins.
def report(name, fit, proba):
    t0 = time.perf_counter()
    model = fit()
    dt = time.perf_counter() - t0
    auc = roc_auc_score(yte, proba(model))
    print(f"{name:>10}: fit {dt:7.2f}s   test AUC {auc:.4f}")

report("macboost",
       lambda: MacBoostClassifier(n_estimators=100, max_depth=6).fit(Xtr, ytr),
       lambda m: m.predict_proba(Xte)[:, 1])

report("mb+goss",
       lambda: MacBoostClassifier(n_estimators=100, max_depth=6, goss=True).fit(Xtr, ytr),
       lambda m: m.predict_proba(Xte)[:, 1])

import lightgbm as lgb  # noqa: E402
report("lightgbm",
       lambda: lgb.LGBMClassifier(n_estimators=100, max_depth=6, num_leaves=64,
                                  learning_rate=0.1, max_bin=255, reg_lambda=1.0,
                                  min_child_samples=1, min_child_weight=1.0,
                                  verbose=-1).fit(Xtr, ytr),
       lambda m: m.predict_proba(Xte)[:, 1])

import xgboost as xgb  # noqa: E402
report("xgboost",
       lambda: xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                 tree_method="hist", max_bin=256, reg_lambda=1.0,
                                 min_child_weight=1.0, verbosity=0).fit(Xtr, ytr),
       lambda m: m.predict_proba(Xte)[:, 1])
