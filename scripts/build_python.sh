#!/bin/bash
# Build the Swift core and bundle the dylib into the Python package.
set -euo pipefail
cd "$(dirname "$0")/.."
swift build -c release
# rm first: cp onto a mapped dylib invalidates its code signature
# in place, and macOS then SIGKILLs every process that loads it.
rm -f python/macboost/libmacboostc.dylib
cp .build/release/libmacboostc.dylib python/macboost/
echo "python package ready: pip install ./python (or set PYTHONPATH=python)"
