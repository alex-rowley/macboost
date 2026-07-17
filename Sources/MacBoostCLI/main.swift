import Foundation
import MacBoost

// macboost — LightGBM-style command line interface.
//
//   macboost train   --data train.csv --label target --output model.json ...
//   macboost predict --model model.json --data test.csv --output preds.csv

let usage = """
macboost — gradient boosted trees on Apple-silicon GPUs

USAGE:
  macboost train --data <train.csv|.tsv|.svm|.mbds> [options]
  macboost predict --model <model.json> --data <test.csv> [options]
  macboost dataset --data <train.csv> --output <train.mbds> [options]

FORMATS:
  CSV / TSV (header row, delimiter sniffed), LibSVM sparse text
  (label idx:val ..., .svm/.libsvm extension or --format libsvm; parsed
  columns are named label,f0,f1,...), and .mbds — MacBoost's binary
  dataset (preprocessed + binned; the `dataset` subcommand or
  --save-dataset creates one, after which training skips parsing and
  binning entirely — ideal for hyperparameter sweeps).

TRAIN OPTIONS:
  --data <path>            training CSV (header row required)          [required]
  --label <name>           label column name                           [default: label]
  --valid <path>           validation CSV (same columns as train)
  --output <path>          model output path                           [default: macboost_model.json]
  --objective <o>          regression | binary | multiclass | mae | huber |
                           quantile | poisson | tweedie               [default: regression]
  --num-classes <n>        class count for multiclass
  --alpha <f>              huber delta / quantile level               [default: 0.9]
  --tweedie-power <f>      tweedie variance power in (1,2)            [default: 1.5]
  --scale-pos-weight <f>   positive-class gradient multiplier (binary)
  --subsample <f>          per-tree row sampling (bagging)            [default: 1.0]
  --feature-fraction <f>   per-tree column sampling                   [default: 1.0]
  --monotone <list>        per-feature constraints, e.g. 0,1,-1,0
  --metric <m>             rmse|mae|logloss|auc|pinball|poisson|tweedie
  --weight-column <name>   sample-weight column in the training file
  --init-model <path>      continue training from a saved model
  --trees <n>              number of boosting rounds                   [default: 100]
  --depth <n>              max tree depth                              [default: 6]
  --leaves <n>             leaf-wise (best-first) growth with this many
                           leaves per tree (LightGBM num_leaves); depth
                           becomes the path cap
  --learning-rate <f>      shrinkage                                   [default: 0.1]
  --lambda <f>             L2 leaf regularisation                      [default: 1.0]
  --min-child-weight <f>   minimum child hessian sum                   [default: 1.0]
  --min-split-gain <f>     minimum gain to split                       [default: 0.0]
  --bins <n>               histogram bins incl. missing bin (4-256)    [default: 256]
  --categorical <cols>     comma-separated categorical column names
                           (values must be integer ids; NaN = missing)
  --goss                   gradient-based one-side sampling (faster training)
  --goss-top-rate <f>      GOSS top-gradient keep fraction              [default: 0.2]
  --goss-other-rate <f>    GOSS uniform sample fraction of the rest     [default: 0.1]
  --feature-selection      Boruta shadow selection before training: only
                           confirmed features may split (GPU-resident)
  --selection-rounds <n>   Boruta rounds                               [default: 20]
  --selection-trees <n>    probe-model rounds            [default: min(trees, 100)]
  --early-stopping <n>     stop after n rounds without valid improvement
  --eval-every <n>         print metrics every n rounds                [default: 10]
  --format <f>             auto | csv | tsv | libsvm                   [default: auto]
  --save-dataset <path>    also write the binned dataset (.mbds) for reuse

PREDICT OPTIONS:
  --model <path>           trained model                               [required]
  --data <path>            input CSV (columns matched by name)         [required]
  --output <path>          predictions output CSV                      [default: predictions.csv]
  --raw                    output raw scores (skip sigmoid for binary)

Missing values: empty CSV fields or non-numeric tokens (NA, nan, ...) are
treated as missing and routed by the learned default direction.
"""

func fail(_ message: String) -> Never {
    FileHandle.standardError.write("error: \(message)\n".data(using: .utf8)!)
    exit(1)
}

func parseFlags(_ args: [String]) -> [String: String] {
    var out: [String: String] = [:]
    var i = 0
    while i < args.count {
        guard args[i].hasPrefix("--") else { fail("unexpected argument '\(args[i])'") }
        let key = String(args[i].dropFirst(2))
        if key == "raw" || key == "goss" || key == "feature-selection" {
            out[key] = "1"; i += 1; continue
        }
        guard i + 1 < args.count else { fail("missing value for --\(key)") }
        out[key] = args[i + 1]
        i += 2
    }
    return out
}

/// Load a text table, choosing the parser by --format or file extension.
func loadTable(_ path: String, flags: [String: String]) throws -> CSVTable {
    let ext = (path as NSString).pathExtension.lowercased()
    let format = flags["format"] ?? "auto"
    switch format {
    case "libsvm": return try CSV.readLibSVM(path: path)
    case "csv", "tsv": return try CSV.read(path: path)
    case "auto":
        return ext == "svm" || ext == "libsvm"
            ? try CSV.readLibSVM(path: path)
            : try CSV.read(path: path)
    case let f: fail("unknown --format '\(f)' (auto | csv | tsv | libsvm)")
    }
}

/// Shared text-table -> (X, y, names, categorical indices) preparation.
func prepare(_ table: CSVTable, flags: [String: String])
    -> (X: [Float], y: [Float], names: [String], categorical: Set<Int>) {
    let labelName = flags["label"] ?? "label"
    guard let labelCol = table.columnIndex(labelName) else {
        fail("label column '\(labelName)' not found; columns: \(table.header.joined(separator: ", "))")
    }
    let featureCols = (0..<table.cols).filter { $0 != labelCol }
    let featureNames = featureCols.map { table.header[$0] }
    var categorical = Set<Int>()
    if let catList = flags["categorical"] {
        for name in catList.split(separator: ",").map(String.init) {
            guard let idx = featureNames.firstIndex(of: name) else {
                fail("categorical column '\(name)' not found among features")
            }
            categorical.insert(idx)
        }
    }
    return (table.featureMajor(columns: featureCols), table.column(labelCol),
            featureNames, categorical)
}

func runDataset(_ flags: [String: String]) {
    guard let dataPath = flags["data"] else { fail("--data is required") }
    guard let outputPath = flags["output"] else { fail("--output is required (e.g. train.mbds)") }
    var params = BoosterParams()
    params.numBins = Int(flags["bins"] ?? "256") ?? 256
    do {
        let t0 = Date()
        let table = try loadTable(dataPath, flags: flags)
        let (X, y, names, categorical) = prepare(table, flags: flags)
        params.categoricalFeatures = categorical
        let booster = try MacBooster(params: params)
        let ds = try booster.makeDataset(featureMajor: X, rows: table.rows,
                                         cols: names.count, labels: y,
                                         featureNames: names)
        try ds.save(to: URL(fileURLWithPath: outputPath))
        print(String(format: "binned %d rows x %d features into %@ in %.2fs",
                     table.rows, names.count, outputPath, -t0.timeIntervalSinceNow))
    } catch {
        fail("\(error)")
    }
}

func runTrain(_ flags: [String: String]) {
    guard let dataPath = flags["data"] else { fail("--data is required (see macboost --help)") }
    let outputPath = flags["output"] ?? "macboost_model.json"

    var params = BoosterParams()
    params.numTrees = Int(flags["trees"] ?? "100") ?? 100
    params.maxDepth = Int(flags["depth"] ?? "6") ?? 6
    if let l = flags["leaves"].flatMap({ Int($0) }) {
        params.numLeaves = l
        if flags["depth"] == nil { params.maxDepth = 12 }   // cap, not shape
    }
    params.learningRate = Float(flags["learning-rate"] ?? "0.1") ?? 0.1
    params.lambda = Float(flags["lambda"] ?? "1.0") ?? 1.0
    params.minChildHess = Float(flags["min-child-weight"] ?? "1.0") ?? 1.0
    params.minSplitGain = Float(flags["min-split-gain"] ?? "0") ?? 0
    params.numBins = Int(flags["bins"] ?? "256") ?? 256
    switch flags["objective"] ?? "regression" {
    case "regression", "l2": params.objective = .regression
    case "binary": params.objective = .binaryLogistic
    case "mae", "l1": params.objective = .mae
    case "huber": params.objective = .huber
    case "quantile": params.objective = .quantile
    case "poisson": params.objective = .poisson
    case "tweedie": params.objective = .tweedie
    case "multiclass": params.objective = .multiclass
    case let o: fail("unknown objective '\(o)'")
    }
    params.numClasses = Int(flags["num-classes"] ?? "1") ?? 1
    params.alpha = Float(flags["alpha"] ?? "0.9") ?? 0.9
    params.tweedieVariancePower = Float(flags["tweedie-power"] ?? "1.5") ?? 1.5
    params.scalePosWeight = Float(flags["scale-pos-weight"] ?? "1") ?? 1
    params.subsample = Float(flags["subsample"] ?? "1") ?? 1
    params.featureFraction = Float(flags["feature-fraction"] ?? "1") ?? 1
    if let m = flags["metric"], let em = EvalMetric(rawValue: m) { params.metric = em }
    if flags["goss"] != nil {
        params.goss = true
        params.gossTopRate = Float(flags["goss-top-rate"] ?? "0.2") ?? 0.2
        params.gossOtherRate = Float(flags["goss-other-rate"] ?? "0.1") ?? 0.1
    }
    let earlyStopping = Int(flags["early-stopping"] ?? "0") ?? 0
    let evalEvery = Int(flags["eval-every"] ?? "10") ?? 10

    do {
        let t0 = Date()
        let booster: MacBooster
        let timings: FitTimings

        /// Parse a text validation file, mapping columns by name.
        func loadValid(featureNames: [String]) throws -> EvalSet? {
            guard let validPath = flags["valid"] else { return nil }
            let vt = try loadTable(validPath, flags: flags)
            let labelName = flags["label"] ?? "label"
            guard let labelCol = vt.columnIndex(labelName) else {
                fail("label column '\(labelName)' not found in validation file")
            }
            let columns = featureNames.map { name -> Int in
                guard let idx = vt.columnIndex(name) else {
                    fail("validation file is missing column '\(name)'")
                }
                return idx
            }
            return EvalSet(featureMajor: vt.featureMajor(columns: columns),
                           rows: vt.rows, labels: vt.column(labelCol))
        }

        if dataPath.hasSuffix(".mbds") {
            let ds = try BinnedDataset.load(from: URL(fileURLWithPath: dataPath))
            print("loaded binned dataset: \(ds.rows) rows x \(ds.cols) features " +
                  String(format: "in %.2fs", -t0.timeIntervalSinceNow))
            let evalSet = try loadValid(featureNames: ds.featureNames
                ?? (0..<ds.cols).map { "f\($0)" })
            booster = try MacBooster(params: params)
            print("training on \(booster.deviceName)")
            timings = try booster.fit(dataset: ds, valid: evalSet,
                                      earlyStoppingRounds: earlyStopping,
                                      evalEvery: evalEvery) { print($0) }
        } else {
            let table = try loadTable(dataPath, flags: flags)
            var (X, y, featureNames, categorical) = prepare(table, flags: flags)
            params.categoricalFeatures = categorical
            var weights: [Float]?
            if let wcol = flags["weight-column"] {
                guard let wi = featureNames.firstIndex(of: wcol) else {
                    fail("weight column '\(wcol)' not found among features")
                }
                // Remove the weight column from the features.
                let stride = table.rows
                weights = Array(X[(wi * stride)..<((wi + 1) * stride)])
                var X2 = [Float](); X2.reserveCapacity((featureNames.count - 1) * stride)
                for f in 0..<featureNames.count where f != wi {
                    X2.append(contentsOf: X[(f * stride)..<((f + 1) * stride)])
                }
                X = X2
                featureNames.remove(at: wi)
                params.categoricalFeatures = Set(categorical.compactMap {
                    $0 == wi ? nil : ($0 > wi ? $0 - 1 : $0)
                })
            }
            if let mono = flags["monotone"] {
                params.monotoneConstraints = mono.split(separator: ",").map { Int($0) ?? 0 }
            }
            let initModel = try flags["init-model"].map {
                try MacBooster.load(from: URL(fileURLWithPath: $0))
            }
            let evalSet = try loadValid(featureNames: featureNames)
            print("loaded \(table.rows) rows x \(featureNames.count) features " +
                  String(format: "in %.2fs", -t0.timeIntervalSinceNow))
            if flags["feature-selection"] != nil {
                let selector = try MacBooster(params: params)
                let rounds = Int(flags["selection-rounds"] ?? "20") ?? 20
                let selTrees = flags["selection-trees"].flatMap { Int($0) }
                let sel = try selector.selectFeatures(
                    featureMajor: X, rows: table.rows, cols: featureNames.count,
                    labels: y, weights: weights, rounds: rounds,
                    trees: selTrees) { print($0) }
                for f in sel.rejected {
                    print("  rejected: \(featureNames[f]) " +
                          String(format: "(beat shadows %d/%d rounds)", sel.hits[f], rounds))
                }
                let keep = sel.confirmed.isEmpty
                    ? sel.confirmed + sel.tentative : sel.confirmed
                guard !keep.isEmpty else {
                    fail("feature selection rejected every feature; no signal found")
                }
                print("keeping \(keep.count)/\(featureNames.count) features: " +
                      keep.map { featureNames[$0] }.joined(separator: ", "))
                params.allowedFeatures = Set(keep)
            }
            booster = try MacBooster(params: params)
            print("training on \(booster.deviceName)")
            if let dsPath = flags["save-dataset"] {
                let ds = try booster.makeDataset(featureMajor: X, rows: table.rows,
                                                 cols: featureNames.count, labels: y,
                                                 featureNames: featureNames)
                try ds.save(to: URL(fileURLWithPath: dsPath))
                print("binned dataset saved to \(dsPath)")
                timings = try booster.fit(dataset: ds, valid: evalSet,
                                          earlyStoppingRounds: earlyStopping,
                                          evalEvery: evalEvery) { print($0) }
            } else {
                timings = try booster.fit(featureMajor: X, rows: table.rows,
                                          cols: featureNames.count, labels: y,
                                          weights: weights,
                                          valid: evalSet, earlyStoppingRounds: earlyStopping,
                                          evalEvery: evalEvery, initModel: initModel) { print($0) }
                booster.featureNames = featureNames
            }
        }
        try booster.save(to: URL(fileURLWithPath: outputPath))
        print(String(format: "trained %d trees in %.2fs (%.1f trees/s)",
                     booster.trees.count, timings.total,
                     Double(booster.trees.count) / timings.total))
        if let best = booster.bestIteration {
            print("best iteration: \(best)")
        }
        print("model saved to \(outputPath)")
    } catch {
        fail("\(error)")
    }
}

func runPredict(_ flags: [String: String]) {
    guard let modelPath = flags["model"] else { fail("--model is required") }
    guard let dataPath = flags["data"] else { fail("--data is required") }
    let outputPath = flags["output"] ?? "predictions.csv"

    do {
        let booster = try MacBooster.load(from: URL(fileURLWithPath: modelPath))
        let table = try CSV.read(path: dataPath)

        // Map model features to CSV columns by name when available,
        // otherwise take the first numFeatures columns in order.
        let columns: [Int]
        if let names = booster.featureNames {
            columns = try names.map { name in
                guard let idx = table.columnIndex(name) else {
                    throw CSVError.malformed("column '\(name)' required by the model is missing")
                }
                return idx
            }
        } else {
            guard table.cols >= booster.numFeatures else {
                fail("model expects \(booster.numFeatures) features, CSV has \(table.cols) columns")
            }
            columns = Array(0..<booster.numFeatures)
        }

        let X = table.featureMajor(columns: columns)
        var preds = booster.predict(featureMajor: X, rows: table.rows, cols: columns.count)
        let K = booster.trainedNumClasses
        if K > 1 {
            if flags["raw"] == nil {          // softmax per row
                for i in 0..<table.rows {
                    var mx = -Float.infinity
                    for c in 0..<K { mx = max(mx, preds[i * K + c]) }
                    var sum: Float = 0
                    for c in 0..<K { preds[i * K + c] = exp(preds[i * K + c] - mx); sum += preds[i * K + c] }
                    for c in 0..<K { preds[i * K + c] /= sum }
                }
            }
            var s = (0..<K).map { "class_\($0)" }.joined(separator: ",") + "\n"
            for i in 0..<table.rows {
                s += (0..<K).map { "\(preds[i * K + $0])" }.joined(separator: ",") + "\n"
            }
            try s.write(toFile: outputPath, atomically: true, encoding: .utf8)
        } else {
            if booster.params.objective == .binaryLogistic && flags["raw"] == nil {
                for i in 0..<preds.count { preds[i] = 1 / (1 + exp(-preds[i])) }
            }
            try CSV.writePredictions(preds, to: outputPath, header: "prediction")
        }
        print("wrote \(table.rows) predictions to \(outputPath)")
    } catch {
        fail("\(error)")
    }
}

let argv = Array(CommandLine.arguments.dropFirst())
switch argv.first {
case "train":
    runTrain(parseFlags(Array(argv.dropFirst())))
case "predict":
    runPredict(parseFlags(Array(argv.dropFirst())))
case "dataset":
    runDataset(parseFlags(Array(argv.dropFirst())))
case "importance":
    let flags = parseFlags(Array(argv.dropFirst()))
    guard let modelPath = flags["model"] else { fail("--model is required") }
    do {
        let b = try MacBooster.load(from: URL(fileURLWithPath: modelPath))
        let gain = b.featureImportance(type: .gain)
        let split = b.featureImportance(type: .split)
        let names = b.featureNames ?? (0..<b.numFeatures).map { "f\($0)" }
        func pad(_ s: String, _ w: Int) -> String {
            s.count >= w ? s : s + String(repeating: " ", count: w - s.count)
        }
        print(pad("feature", 24) + pad("gain", 14) + "splits")
        for i in gain.indices.sorted(by: { gain[$0] > gain[$1] }) {
            print(pad(names[i], 24)
                  + pad(String(format: "%.1f", gain[i]), 14)
                  + String(format: "%.0f", split[i]))
        }
    } catch { fail("\(error)") }
case "--version", "version":
    print("macboost \(macBoostVersion)")
case "--help", "-h", "help", nil:
    print(usage)
case let cmd?:
    fail("unknown command '\(cmd)' — expected train or predict (see macboost --help)")
}
