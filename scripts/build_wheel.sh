#!/bin/bash
# Build the Python wheel. Bundles the Metal dylib but keeps the default
# py3-none-any tag: training/native inference needs the dylib (Apple
# silicon), while pure-Python inference and the export tools work anywhere,
# so the wheel must be installable on Linux too.
# Uses uv when available (local dev), pip otherwise (CI with setup-python).
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release
# rm first: cp onto a mapped dylib invalidates its code signature
# in place, and macOS then SIGKILLs every process that loads it.
rm -f python/macboost/libmacboostc.dylib
cp .build/release/libmacboostc.dylib python/macboost/
cd python
rm -rf dist build ./*.egg-info
if command -v uv >/dev/null 2>&1; then
  uv build --wheel --out-dir dist .
else
  python3 -m pip install --quiet --upgrade build
  python3 -m build --wheel --outdir dist .
fi
ls -la dist/
