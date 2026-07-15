# macboost (Python)

Python bindings for [MacBoost](../README.md) — gradient boosted trees on
Apple-silicon GPUs, with a scikit-learn-style API (`MacBoostRegressor`,
`MacBoostClassifier`).

Build the native core first, then install:

```bash
swift build -c release && ../scripts/build_python.sh   # bundles the dylib
pip install .
```

See the repository README for usage and benchmarks.
