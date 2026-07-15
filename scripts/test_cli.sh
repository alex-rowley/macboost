#!/bin/bash
# End-to-end CLI smoke test: generate CSV data (with missing values and a
# categorical column), train with a validation set + early stopping, predict,
# and assert the validation RMSE beats a threshold.
set -euo pipefail
cd "$(dirname "$0")/.."

swift build -c release 2>/dev/null
CLI=.build/release/macboost
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

python3 - "$WORK" << 'EOF'
import csv, math, random, sys
work = sys.argv[1]
random.seed(7)

def make(path, n):
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["f0", "f1", "f2", "cat", "target"])
        for _ in range(n):
            x0, x1, x2 = (random.random() for _ in range(3))
            cat = random.randrange(4)
            y = (10 * math.sin(3.14159 * x0 * x1) + 20 * (x2 - 0.5) ** 2
                 + (3 if cat in (0, 3) else -3) + random.gauss(0, 1))
            row = [f"{x0:.6f}", f"{x1:.6f}", f"{x2:.6f}", cat, f"{y:.6f}"]
            if random.random() < 0.1:
                row[0] = ""          # missing value
            w.writerow(row)

make(f"{work}/train.csv", 20000)
make(f"{work}/valid.csv", 5000)
EOF

$CLI train --data "$WORK/train.csv" --valid "$WORK/valid.csv" --label target \
    --categorical cat --trees 300 --depth 5 --early-stopping 30 --eval-every 50 \
    --output "$WORK/model.json"

$CLI predict --model "$WORK/model.json" --data "$WORK/valid.csv" \
    --output "$WORK/preds.csv"

python3 - "$WORK" << 'EOF'
import csv, math, sys
work = sys.argv[1]
with open(f"{work}/valid.csv") as fh:
    y = [float(r["target"]) for r in csv.DictReader(fh)]
with open(f"{work}/preds.csv") as fh:
    p = [float(r["prediction"]) for r in csv.DictReader(fh)]
assert len(y) == len(p), f"row mismatch: {len(y)} vs {len(p)}"
rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y, p)) / len(y))
baseline = math.sqrt(sum((a - sum(y)/len(y)) ** 2 for a in y) / len(y))
print(f"CLI valid RMSE {rmse:.4f} (baseline {baseline:.4f})")
assert rmse < 1.4, f"RMSE {rmse} above 1.4 floor (noise sigma 1.0)"
EOF

# --- TSV: same pipeline through the tab-delimited path -------------------
python3 - "$WORK" << 'EOF'
import sys
work = sys.argv[1]
for name in ("train", "valid"):
    with open(f"{work}/{name}.csv") as src, open(f"{work}/{name}.tsv", "w") as dst:
        for line in src:
            dst.write(line.replace(",", "\t"))
EOF
$CLI train --data "$WORK/train.tsv" --valid "$WORK/valid.tsv" --label target \
    --categorical cat --trees 100 --depth 5 --eval-every 100 \
    --output "$WORK/model_tsv.json" | grep -q "trained 100 trees" \
    || { echo "TSV training failed"; exit 1; }
echo "TSV path OK"

# --- LibSVM: sparse text with 1-based indices -----------------------------
python3 - "$WORK" << 'EOF'
import csv, sys
work = sys.argv[1]
with open(f"{work}/train.csv") as src, open(f"{work}/train.svm", "w") as dst:
    for r in csv.DictReader(src):
        pairs = [f"{i+1}:{r[c]}" for i, c in enumerate(("f0", "f1", "f2", "cat")) if r[c] != ""]
        dst.write(f"{r['target']} " + " ".join(pairs) + "\n")
EOF
$CLI train --data "$WORK/train.svm" --trees 100 --depth 5 --eval-every 100 \
    --output "$WORK/model_svm.json" | grep -q "trained 100 trees" \
    || { echo "LibSVM training failed"; exit 1; }
echo "LibSVM path OK"

# --- Binned dataset (.mbds): build once, retrain from it ------------------
$CLI dataset --data "$WORK/train.csv" --label target --categorical cat \
    --output "$WORK/train.mbds" | grep -q "binned" \
    || { echo "dataset subcommand failed"; exit 1; }
$CLI train --data "$WORK/train.mbds" --valid "$WORK/valid.csv" --label target \
    --trees 300 --depth 5 --early-stopping 30 --eval-every 100 \
    --output "$WORK/model_mbds.json" > "$WORK/mbds_log.txt"
grep -q "loaded binned dataset" "$WORK/mbds_log.txt" \
    || { echo "mbds training failed"; cat "$WORK/mbds_log.txt"; exit 1; }
$CLI predict --model "$WORK/model_mbds.json" --data "$WORK/valid.csv" \
    --output "$WORK/preds_mbds.csv"
python3 - "$WORK" << 'EOF'
import csv, math, sys
work = sys.argv[1]
with open(f"{work}/valid.csv") as fh:
    y = [float(r["target"]) for r in csv.DictReader(fh)]
with open(f"{work}/preds_mbds.csv") as fh:
    p = [float(r["prediction"]) for r in csv.DictReader(fh)]
rmse = math.sqrt(sum((a - b) ** 2 for a, b in zip(y, p)) / len(y))
print(f"mbds-trained valid RMSE {rmse:.4f}")
assert rmse < 1.4, f"mbds model RMSE {rmse} above floor"
EOF
echo "mbds path OK"

# --- Multiclass + importance ----------------------------------------------
python3 - "$WORK" << 'PYEOF2'
import csv, random, sys
work = sys.argv[1]
random.seed(9)
with open(f"{work}/mc.csv", "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["f0", "f1", "target"])
    for _ in range(8000):
        x0, x1 = random.random(), random.random()
        cls = (1 if x0 > 0.5 else 0) + (2 if x1 > 0.5 else 0)
        w.writerow([f"{x0:.5f}", f"{x1:.5f}", cls])
PYEOF2
$CLI train --data "$WORK/mc.csv" --label target --objective multiclass \
    --num-classes 4 --trees 40 --depth 4 --eval-every 100 \
    --output "$WORK/mc.json" | grep -q "trained 160 trees" \
    || { echo "multiclass training failed"; exit 1; }
$CLI predict --model "$WORK/mc.json" --data "$WORK/mc.csv" --output "$WORK/mc_preds.csv"
head -1 "$WORK/mc_preds.csv" | grep -q "class_0,class_1,class_2,class_3" \
    || { echo "multiclass predict columns wrong"; exit 1; }
python3 - "$WORK" << 'PYEOF2'
import csv, sys
work = sys.argv[1]
with open(f"{work}/mc.csv") as fh:
    ys = [int(r["target"]) for r in csv.DictReader(fh)]
correct = 0
with open(f"{work}/mc_preds.csv") as fh:
    for y, r in zip(ys, csv.DictReader(fh)):
        probs = [float(r[f"class_{c}"]) for c in range(4)]
        assert abs(sum(probs) - 1) < 1e-3, "probs must sum to 1"
        if probs.index(max(probs)) == y: correct += 1
acc = correct / len(ys)
print(f"multiclass CLI accuracy {acc:.3f}")
assert acc > 0.95, acc
PYEOF2
echo "multiclass path OK"

$CLI importance --model "$WORK/model.json" | head -3 | grep -q "feature" \
    || { echo "importance subcommand failed"; exit 1; }
echo "importance OK"

echo "CLI smoke test PASSED"
