#!/bin/bash
# Build the Swift core and bundle the dylib into the Python package.
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release
cp .build/release/libmacboostc.dylib python/macboost/
echo "python package ready: pip install ./python (or set PYTHONPATH=python)"
