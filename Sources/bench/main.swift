import Foundation
import MacBoost

// Simple --key value argument parsing.
func parseArgs() -> [String: String] {
    var out: [String: String] = [:]
    let argv = CommandLine.arguments
    var i = 1
    while i < argv.count {
        if argv[i].hasPrefix("--"), i + 1 < argv.count {
            out[String(argv[i].dropFirst(2))] = argv[i + 1]
            i += 2
        } else {
            i += 1
        }
    }
    return out
}

func rmse(_ a: [Float], _ b: [Float]) -> Float {
    var s = 0.0
    for i in 0..<a.count { let d = Double(a[i] - b[i]); s += d * d }
    return Float((s / Double(a.count)).squareRoot())
}

func writeRaw(_ array: [Float], to url: URL) throws {
    try array.withUnsafeBytes { raw in
        try Data(bytesNoCopy: UnsafeMutableRawPointer(mutating: raw.baseAddress!),
                 count: raw.count, deallocator: .none).write(to: url)
    }
}

let a = parseArgs()
let rows = Int(a["rows"] ?? "1000000")!
let cols = Int(a["cols"] ?? "100")!
let validRows = Int(a["valid"] ?? "200000")!
let numTrees = Int(a["trees"] ?? "100")!
let maxDepth = Int(a["depth"] ?? "6")!
let lr = Float(a["lr"] ?? "0.1")!
let evalEvery = Int(a["eval-every"] ?? "20")!
let seed = UInt64(a["seed"] ?? "42")!
let numCats = Int(a["cats"] ?? "0")!
let missingFrac = Float(a["missing"] ?? "0")!
let evalValid = (a["eval-valid"] ?? "0") == "1"
let goss = (a["goss"] ?? "0") == "1"
let subsample = Float(a["subsample"] ?? "1")!

let catCards = (0..<numCats).map { [8, 32, 4, 16, 64][$0 % 5] }
let categorical = numCats > 0 ? Set((cols - numCats)..<cols) : Set<Int>()

print("MacBoost bench — rows=\(rows) cols=\(cols) (cats=\(numCats)) valid=\(validRows) " +
      "trees=\(numTrees) depth=\(maxDepth) lr=\(lr) missing=\(missingFrac) " +
      "evalValid=\(evalValid)")

var t0 = DispatchTime.now().uptimeNanoseconds
let (Xtr, ytr): ([Float], [Float])
let (Xva, yva): ([Float], [Float])
if numCats > 0 || missingFrac > 0 {
    let numeric = cols - numCats
    (Xtr, ytr) = DataGen.friedmanMixed(rows: rows, numericCols: numeric, catCards: catCards,
                                       seed: seed, effectSeed: seed, missingFrac: missingFrac)
    (Xva, yva) = DataGen.friedmanMixed(rows: validRows, numericCols: numeric, catCards: catCards,
                                       seed: seed &+ 999, effectSeed: seed, missingFrac: missingFrac)
} else {
    (Xtr, ytr) = DataGen.friedman(rows: rows, cols: cols, seed: seed)
    (Xva, yva) = DataGen.friedman(rows: validRows, cols: cols, seed: seed &+ 999)
}
print(String(format: "data generated in %.2fs",
             Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9))

if let dir = a["export"] {
    let d = URL(fileURLWithPath: dir, isDirectory: true)
    try FileManager.default.createDirectory(at: d, withIntermediateDirectories: true)
    try writeRaw(Xtr, to: d.appendingPathComponent("X_train.bin"))
    try writeRaw(ytr, to: d.appendingPathComponent("y_train.bin"))
    try writeRaw(Xva, to: d.appendingPathComponent("X_valid.bin"))
    try writeRaw(yva, to: d.appendingPathComponent("y_valid.bin"))
    let meta: [String: Any] = ["train_rows": rows, "valid_rows": validRows, "cols": cols,
                               "layout": "feature_major_float32",
                               "trees": numTrees, "depth": maxDepth, "lr": Double(lr),
                               "categorical": categorical.sorted(),
                               "eval_valid": evalValid]
    let json = try JSONSerialization.data(withJSONObject: meta, options: [.prettyPrinted, .sortedKeys])
    try json.write(to: d.appendingPathComponent("meta.json"))
    print("exported dataset to \(dir)")
}

var params = BoosterParams()
params.numTrees = numTrees
params.maxDepth = maxDepth
params.learningRate = lr
params.categoricalFeatures = categorical
params.goss = goss
params.subsample = subsample

let booster = try MacBooster(params: params)
print("GPU: \(booster.deviceName)")

let evalSet = evalValid ? EvalSet(featureMajor: Xva, rows: validRows, labels: yva) : nil
let timings = try booster.fit(featureMajor: Xtr, rows: rows, cols: cols, labels: ytr,
                              valid: evalSet, evalEvery: evalEvery) { print($0) }

print(String(format: """

    --- timings ---
    binning (edges + GPU bin): %7.3fs
    GPU levels (hist + split): %7.3fs
    CPU split decisions:       %7.3fs
    partition + preds update:  %7.3fs
    total fit:                 %7.3fs  (%.1f trees/s)
    """,
    timings.binning, timings.gpuLevels, timings.cpuDecide, timings.finalize,
    timings.total, Double(numTrees) / timings.total))

t0 = DispatchTime.now().uptimeNanoseconds
let pv = booster.predict(featureMajor: Xva, rows: validRows, cols: cols)
let predTime = Double(DispatchTime.now().uptimeNanoseconds - t0) / 1e9
print(String(format: "valid RMSE: %.5f   (predict %d rows in %.2fs)",
             rmse(pv, yva), validRows, predTime))
