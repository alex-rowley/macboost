# MacBoost

**Gradient boosting at 3.5–11× LightGBM/XGBoost speed, on the GPU already
in your Mac. Same accuracy, same API, zero dependencies.**

If you train gradient boosted trees on Apple silicon today, your GPU sits
idle while LightGBM saturates your CPU cores. MacBoost fixes that: the full
XGBoost/LightGBM histogram algorithm, written from scratch as Metal compute
kernels, with a scikit-learn-compatible Python API, a LightGBM-style CLI,
and a zero-dependency Swift core. On the canonical HIGGS benchmark (10.5M
rows) it trains in **2.8 seconds against LightGBM's 10.3 and XGBoost's
9.3 — at equal AUC** — and up to 11× ahead on wider data.

It is not a wrapper and not a port: no CUDA translation layer, no
PyTorch/MPS, no OpenCL. An entire boosting tree executes as one GPU command
buffer — histograms, split search, and split *decisions* all on-device —
while the host reads finished trees back through unified memory as the GPU
builds the next one.

**Feature parity with LightGBM and XGBoost for single-node regression and
classification on dense data:**

- **Objectives**: L2, L1/MAE, Huber, quantile, Poisson, Tweedie, binary
  logistic (`scale_pos_weight`), multiclass softmax
- **Data**: missing values (learned default directions), native categorical
  features (optimal subset splits), sample weights, CSV/TSV/LibSVM input,
  binary dataset cache (`.mbds`) for instant re-training
- **Growth**: level-wise (default) or LightGBM-style leaf-wise via
  `num_leaves` — best-first splits, monotone/categorical/NaN aware,
  same model format
- **Sampling**: GOSS (LightGBM's algorithm incl. its warm-up), bagging
  (`subsample`), column sampling (`colsample_bytree`)
- **Constraints & interpretability**: monotone constraints with hard
  bounds-propagated guarantees, gain/split feature importance, GPU
  TreeSHAP contributions (incl. multiclass; ~50M row·trees/s — 200k rows
  × 100 trees explained in 0.4s), calibrated L1/quantile leaves
- **Feature selection**: built-in Boruta (shadow features) — see below;
  no other GBM library ships this natively
- **Workflow**: early stopping with per-iteration GPU validation eval,
  metric overrides incl. AUC, warm starts (`init_model`), JSON model
  save/load, GPU batch inference (bit-identical to the CPU path)

And a few things the incumbents don't have: bit-exact GPU/CPU inference,
target-leakage detection at fit time, and typed input errors with row
numbers instead of a segfault. MIT licensed.

## Benchmarks

All numbers from an M4 Max (40-core GPU), macOS 26. LightGBM 4.x and
XGBoost 3.x (`tree_method="hist"`) use all 16 CPU cores. Identical data —
byte-exact exported matrices — and matched hyperparameters: 100 trees,
depth 6, learning rate 0.1, 255/256 bins. Accuracy sits next to every
timing, because speed at degraded accuracy is worthless.

### Real data: HIGGS (UCI, 10.5M rows × 28 features, binary)

The canonical GBDT benchmark from the original XGBoost and LightGBM
papers, standard protocol (last 500k rows as test):

| library | fit time | test AUC |
|---|---|---|
| **MacBoost** | **2.8s** | 0.8128 |
| **MacBoost + GOSS** | **1.9s** | 0.8129 |
| XGBoost hist | 9.3s | 0.8128 |
| LightGBM | 10.3s | 0.8125 |

### Synthetic (Friedman #1, known noise floor of 1.0)

| dataset | MacBoost | LightGBM | XGBoost hist |
|---|---|---|---|
| 1M × 100, numeric | **0.49s** / RMSE 1.038 | 3.76s / 1.038 | 3.25s / 1.039 |
| 1M × 100, 20% NaN + 20 categorical cols, per-iteration valid eval | **0.76s** / 2.493 | 4.03s / 2.493 | 7.67s / 2.497 |
| 10M × 100, numeric | **3.9s** / 1.033 | 27.5s / 1.034 | 31.5s / 1.033 |
| 20k × 100 (small-data throughput) | **947 trees/s** | — | — |

GOSS (off by default, as in LightGBM) adds another ~1.5–1.8×: 1M in 0.36s,
10M in 2.0s, at ≤0.004 RMSE cost. Inference is GPU-accelerated too: 500k
rows through 100 trees in 0.03s.

**Small datasets: use fewer bins.** Histogram maintenance scales with
`max_bin` regardless of row count, while split resolution stops improving
once bins exceed your rows-per-node — so below ~100k rows, `max_bin=64`
(or 32) is typically 2–4× faster with identical accuracy. `fit` warns
when your configuration is in this territory.

**Reproduce:**

```sh
uv run scripts/bench_higgs.py                 # downloads HIGGS on first run
swift run -c release bench --rows 1000000 --cols 100 --export export
uv run scripts/compare.py export              # LightGBM + XGBoost, same bytes
```

Why the speedup varies: histogram work — where the GPU dominates — scales
with rows × features, so wide data (100 columns: 5–8×) benefits more than
narrow data (HIGGS's 28 columns: 3.5×). Categorical-heavy workloads widen
the gap further because subset-split search is nearly free in-kernel.

## Install

Training requires an Apple-silicon Mac on macOS 15+. The wheel itself
installs on any platform — on Linux you get inference and model export,
not training (see [Deployment](#deployment)).

**Python** — a prebuilt wheel, no compiler or toolchain needed:

```sh
pip install macboost
```

**CLI** — download the `macboost` binary from GitHub Releases
(`curl -LO`, not the browser, to avoid the quarantine prompt), or build
from source.

**Swift** — add the package to your `Package.swift`:

```swift
.package(url: "https://github.com/alex-rowley/macboost", from: "0.1.0")
```

**From source** (needs Xcode command-line tools; kernels compile at
runtime, so there is no Metal-toolchain download and zero package
dependencies):

```sh
git clone https://github.com/alex-rowley/macboost && cd macboost
swift build -c release        # library, CLI (.build/release/macboost), dylib
swift test                    # 75 behavioural tests
./scripts/build_wheel.sh && pip install python/dist/*.whl
```

## Quick start

### Python

```python
import pandas as pd
from macboost import MacBoostRegressor, MacBoostClassifier

df = pd.read_parquet("claims.parquet")           # any format pandas reads
X = df.drop(columns="loss").to_numpy("float32")  # NaN = missing, handled natively
y = df["loss"].to_numpy("float32")

model = MacBoostRegressor(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    objective="tweedie", tweedie_variance_power=1.5,   # or l2/mae/huber/quantile/poisson
    categorical_features=[3, 7],                        # columns holding integer ids
    monotone_constraints=[0, 1, 0, 0, -1, 0, 0, 0],
    subsample=0.8, colsample_bytree=0.8,
)
model.fit(X, y, eval_set=(X_val, y_val), early_stopping_rounds=50,
          sample_weight=exposure)

preds = model.predict(X_test)                    # GPU for large batches
model.feature_importances_                       # gain-based; .feature_importance("split")
model.predict_contrib(X_test)                    # TreeSHAP, rows sum to prediction
model.save_model("model.json")
```

`MacBoostClassifier` handles binary and multiclass automatically (labels
encoded like sklearn — strings work), with `predict_proba`,
`scale_pos_weight`, and AUC-driven early stopping (`eval_metric="auc"`).
The estimators implement the scikit-learn protocol, so `clone`,
`Pipeline`, `GridSearchCV`, `cross_val_score` and Optuna work out of the
box — sklearn itself remains an optional dependency.

**Coming from LightGBM or XGBoost?** Parameters use the dialect their
sklearn wrappers share, and the well-known native spellings are accepted
as aliases:

| macboost | LightGBM also accepts | XGBoost also accepts |
|---|---|---|
| `n_estimators` | `num_iterations`, `num_trees` | `num_round`, `num_boost_round` |
| `learning_rate` | `eta`, `shrinkage_rate` | `eta` |
| `reg_lambda` | `lambda_l2` | `lambda` (use `reg_lambda`) |
| `min_split_gain` | `min_gain_to_split` | `gamma`, `min_split_loss` |
| `min_child_weight` | `min_sum_hessian_in_leaf` | `min_child_weight` |
| `max_bin` | `max_bin` (macboost's count includes the reserved missing bin) | `max_bin` |
| `num_leaves` | `num_leaves` (enables best-first growth; `max_depth` caps the path) | `max_leaves` |
| `categorical_features` | `categorical_feature` | (dtype-based) |
| `subsample` | `bagging_fraction` | `subsample` |
| `colsample_bytree` | `feature_fraction` | `colsample_bytree` |
| `goss`, `goss_top_rate`, `goss_other_rate` | `data_sample_strategy=goss`, `top_rate`, `other_rate` | — |
| `objective`, `alpha`, `tweedie_variance_power`, `scale_pos_weight`, `monotone_constraints`, `metric`/`eval_metric` | same | same |

`early_stopping_rounds` works in the constructor (XGBoost 2.x style) or in
`fit()` (classic style); `eval_set` takes a tuple or a one-element list.

### CLI

```sh
macboost train --data train.csv --label target --valid valid.csv \
    --objective quantile --alpha 0.9 --categorical store_id,region \
    --monotone 0,1,0,0 --trees 500 --early-stopping 50 --output model.json

macboost predict --model model.json --data test.csv --output preds.csv
macboost importance --model model.json

# Hyperparameter sweeps: bin once, retrain instantly (save_binary pattern):
macboost dataset --data train.csv --label target --output train.mbds
macboost train --data train.mbds --trees 1000 --learning-rate 0.05 ...
```

Inputs: CSV/TSV (header row, delimiter sniffed; empty or non-numeric
fields are missing values), LibSVM sparse text (`label idx:val ...`), and
`.mbds` binned datasets. Sample weights via `--weight-column`, warm starts
via `--init-model`; multiclass predictions come out as per-class
probability columns. `macboost --help` lists everything.

### Swift

```swift
import MacBoost

var params = BoosterParams()
params.numTrees = 500
params.objective = .poisson
params.monotoneConstraints = [0, 1, 0, 0]
let booster = try MacBooster(params: params)
try booster.fit(featureMajor: X, rows: n, cols: f, labels: y,
                weights: w,
                valid: EvalSet(featureMajor: Xv, rows: v, labels: yv),
                earlyStoppingRounds: 50)
let preds = booster.predict(featureMajor: Xtest, rows: m, cols: f)
let shap = booster.predictContributions(featureMajor: Xtest, rows: m, cols: f)
try booster.save(to: url)
```

The library target has zero dependencies beyond Apple's OS frameworks.

## Deployment

Models train on a Mac; they usually ship to Linux. Three paths, all
producing identical predictions (verified in the test suite):

**1. `pip install macboost` on the server.** The wheel installs anywhere.
Off Apple silicon there is no Metal core, so training raises, but
`load_model` falls back to a pure-numpy scorer with the same semantics —
NaN routing, categorical splits, objective transforms:

```python
model = MacBoostRegressor.load_model("model.json")   # works on Linux
preds = model.predict(X)
```

Simplest path; fine for batch scoring. It is numpy-speed, not
XGBoost-speed — for latency-critical serving use one of the exports.

**2. XGBoost export.** Emit a standard XGBoost JSON model and serve it
with vanilla `xgboost` (or anything that loads one — e.g. Triton FIL):

```python
model.save_xgboost("model.xgb.json")
# elsewhere: bst = xgb.Booster(); bst.load_model("model.xgb.json")
```

Categorical splits have no XGBoost equivalent, so they are rewritten into
equivalent numeric-split chains — predictions match to float32 epsilon,
but exported trees are larger and per-tree introspection won't mirror the
original.

**3. ONNX export.** Emit an `ai.onnx.ml` TreeEnsemble for onnxruntime
(requires the `onnx` package to export, only onnxruntime to serve):

```python
model.save_onnx("model.onnx")
# elsewhere: ort.InferenceSession("model.onnx").run(["prediction"], {"input": X})
```

The graph outputs what `predict` returns: probabilities for classifiers,
means for poisson/tweedie, raw scores for regression.

## Feature selection (Boruta, GPU-resident)

Everyone benchmarks feature importance against random probes eventually;
the principled version is **Boruta** (Kursa & Rudnicki 2010): train on
`[X | shadow(X)]` where shadows are row-permuted copies of the real
columns, score a feature a "hit" each round its gain beats the *best*
shadow, repeat with fresh permutations, and let a binomial test sort
features into confirmed / tentative / rejected. Nobody ships it natively
because it retrains the model ~20 times — which costs minutes elsewhere
and seconds here.

```python
model = MacBoostRegressor(feature_selection=True).fit(X, y)   # select, then train clean
model.selected_features_          # surviving column indices
model.selection_result_           # hits, verdicts, gain vs shadow ceiling

sel = MacBoostRegressor().select_features(X, y, rounds=20)    # selection only
sel.confirmed_, sel.tentative_, sel.rejected_
```

The disposable probe models default to `min(n_estimators, 100)` boosting
rounds — Boruta needs gain-vs-shadow votes, not converged ensembles. Tune
with `selection_estimators=` (40–50 is the cheapest sound setting).

CLI: `macboost train --feature-selection [--selection-rounds 20] ...`

Implementation is unique to this engine: shadows never exist as data.
The training matrix is already binned in GPU memory, and a permuted
column has identical bin edges to its original — so a kernel gathers
each column's bin bytes through a per-column random bijection (a Feistel
network, no sort, no index array) straight into a double-width binned
matrix that lives only on the GPU for the duration. Each round re-seeds
the permutation and retrains; your `X` is never copied, extended, or
touched. The final model trains with rejected features masked out of the
split search, so it still accepts full-width `X` at predict time — no
column bookkeeping downstream.

## Guardrails

Input mistakes fail loudly with typed errors and row numbers: shape
mismatches, NaN/Inf labels, non-{0,1} binary labels, continuous labels fed
to the classifier, out-of-range parameters or category ids, wrong predict
width. A subsample correlation check warns (`UserWarning` in Python,
`fitWarnings` in Swift) when a feature is nearly identical to the label —
the classic "target or prediction column left inside X" leakage bug that
LightGBM and XGBoost train on silently.

## How it works

Classic histogram GBDT (level-wise or best-first growth, second-order
gain), where every
stage is a Metal kernel and **an entire tree is one GPU command buffer** —
`decide_splits` picks splits on-device and writes the *next* level's
dispatch arguments (indirect dispatch), so training has zero per-level CPU
round-trips. Trees are pipelined: the host reads tree *t* back through
unified memory while the GPU builds tree *t+1*.

The hot loop is tuned for Apple-silicon memory bandwidth:

- **Quantised gradients**: per tree, gradients scale to 16 bits and
  hessians to 7 bits against GPU-reduced maxima, packed into one 32-bit
  word per sample. Threadgroup histogram accumulation is 2 native integer
  atomics per (sample, feature) — Metal forbids threadgroup float atomics,
  and this beats the bit-cast CAS workaround. Device histograms stay
  float32, so split search is precision-agnostic (the hand-computed
  exact-split tests pass under quantisation).
- **Feature-tiled bin matrix**: 8 features per fetch in a sample-major
  tile layout — each sample's order index and gradient word are read once
  per 8 features instead of once per feature.
- **Histogram subtraction** (LightGBM's trick): only the smaller child of
  each split builds a histogram; the sibling is `parent − child`.
- **Active-row routing**: the per-level sample pass touches only rows whose
  node is still alive; leaf values apply the moment a node dies.
  Threadgroup-aggregated cursor reservation keeps shallow levels from
  serialising on a handful of atomics.
- Everything lives in `storageModeShared` unified memory — model readback,
  validation metrics, and dataset loading never copy across a bus.

Inference and SHAP run on the GPU too: batch prediction walks the
flattened forest over raw values (bit-identical to the CPU path), and
`predict_contrib` uses GPUTreeSHAP-style path decomposition — one thread
per (row, leaf path), duplicate features merged multiplicatively, the
whole Shapley weight computation in registers — for ~40× the parallel CPU
implementation, which remains as the small-batch path and reference.

Missing values get a reserved bin and the split search tries them on both
sides (learned default direction). Categorical features use LightGBM's
Fisher method: bins sorted by gradient ratio in-kernel, best prefix subset
wins, stored as a 256-bit mask per node. Monotone constraints propagate
[lo, hi] bounds down the tree with midpoint caps clamped into the parent's
bounds (the nested-split edge case is covered by a 1,500-point grid-sweep
test). L1/quantile leaves are renewed as per-leaf residual quantiles after
the structure is built, keeping extreme quantiles calibrated (coverage
within ±0.03 of alpha).

## Tests

`swift test` runs 75 behavioural tests adapted from the LightGBM and
XGBoost open-source suites (upstream sources cited per test in
`Tests/MacBoostTests/`):

- **Exact math**: hand-computed 4-point split/leaf values; GPU-chosen
  splits must match an exhaustive CPU brute-force search
- **Accuracy floors calibrated against LightGBM on identical data** and
  against computed Bayes limits — never guessed thresholds
- **Semantic invariants**: quantile coverage equals alpha; weights behave
  identically to row duplication; L1/Huber resist label outliers where L2
  cannot; monotone predictions never decrease along a +1-constrained
  feature (grid sweep plus a structural subtree audit); SHAP rows sum to
  the prediction; GOSS warm-up is exactly full training; tiny-subsample
  degradation proves sampling actually engages
- **Bit-exactness**: GPU inference ≡ CPU inference; save/load round-trips
  to identical predictions; `.mbds`-trained ≡ raw-trained
- End-to-end surfaces: `scripts/test_cli.sh` (CSV/TSV/LibSVM/mbds/
  multiclass/importance flows) and `uv run scripts/test_python.py`
  (35 checks: sklearn interop incl. real GridSearchCV/cross_val_score
  runs, objectives, weights, multiclass with string labels, SHAP,
  guardrails)

These tests caught real bugs during development — a double-applied
learning rate under bagging, a monotonicity leak from crossed bounds in
nested constrained splits, warm-up precision loss under GOSS — which is
exactly what they exist for.

## Scope and non-goals

Parity here means: everything a single-node tabular regression or
classification workflow touches, on dense data. Deliberately **not**
implemented: ranking objectives (LambdaRank), sparse matrix input (native
categoricals cover the dominant one-hot use case), DART, and distributed
training —
a Mac Studio is a single very fast node by design. If one of these gaps
matters to you, the corresponding upstream test suite is the acceptance
bar for a contribution.

## License

MIT — see [LICENSE](LICENSE).
