#!/bin/bash
# Build the platform-tagged Python wheel (bundles the Metal dylib).
# Uses uv when available (local dev), pip otherwise (CI with setup-python).
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release
cp .build/release/libmacboostc.dylib python/macboost/
cd python
rm -rf dist build ./*.egg-info
if command -v uv >/dev/null 2>&1; then
  uv build --wheel --out-dir dist .
  uvx --from wheel wheel tags --platform-tag macosx_15_0_arm64 --remove dist/*.whl
else
  python3 -m pip install --quiet --upgrade build wheel
  python3 -m build --wheel --outdir dist .
  python3 -m wheel tags --platform-tag macosx_15_0_arm64 --remove dist/*.whl
fi
ls -la dist/
