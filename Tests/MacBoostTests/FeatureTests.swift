import XCTest
@testable import MacBoost

/// Tests for missing values, categorical features, early stopping and
/// serialization, adapted from LightGBM's python_package_test suite.
final class FeatureTests: XCTestCase {

    func rmse(_ a: [Float], _ b: [Float]) -> Float {
        var s = 0.0
        for i in 0..<a.count { let d = Double(a[i] - b[i]); s += d * d }
        return Float((s / Double(a.count)).squareRoot())
    }

    // MARK: missing values (LightGBM test_engine.py::test_missing_value_handle)

    /// NaN must be routed by a LEARNED default direction: construct data
    /// where perfect fit requires missing to join the x=0 group, then the
    /// mirrored case where it must join the x=1 group.
    func testMissingValueLearnedDirection() throws {
        for missingJoinsZero in [true, false] {
            var X = [Float](), y = [Float]()
            for _ in 0..<100 { X.append(0); y.append(10) }
            for _ in 0..<100 { X.append(1); y.append(0) }
            for _ in 0..<100 { X.append(.nan); y.append(missingJoinsZero ? 10 : 0) }
            let rows = X.count
            let b = try MacBooster(params: {
                var p = BoosterParams()
                p.numTrees = 30; p.maxDepth = 1; p.learningRate = 0.3
                return p
            }())
            try b.fit(featureMajor: X, rows: rows, cols: 1, labels: y)
            let pred = b.predict(featureMajor: [0, 1, .nan], rows: 3, cols: 1)
            XCTAssertEqual(pred[0], 10, accuracy: 0.2)
            XCTAssertEqual(pred[1], 0, accuracy: 0.2)
            XCTAssertEqual(pred[2], missingJoinsZero ? 10 : 0, accuracy: 0.2,
                           "NaN must follow the learned default direction")
        }
    }

    /// Sprinkling NaN into informative features must degrade gracefully,
    /// not break training.
    func testMissingValuesAccuracy() throws {
        let rows = 30_000, cols = 10
        var (X, y) = DataGen.friedman(rows: rows, cols: cols, seed: 21)
        var rng = SplitMix64(seed: 22)
        for f in 0..<5 {          // 20% missing in the informative features
            for i in 0..<rows where rng.uniform() < 0.2 { X[f * rows + i] = .nan }
        }
        let b = try MacBooster(params: {
            var p = BoosterParams(); p.numTrees = 100; p.maxDepth = 6; return p
        }())
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y)
        let pred = b.predict(featureMajor: X, rows: rows, cols: cols)
        let r = rmse(pred, y)
        let baseline = rmse(y, [Float](repeating: b.baseScore, count: rows))
        // LightGBM scores 2.297 train RMSE on this exact setup (20% NaN in
        // the 5 informative columns raises the 1.0 noise floor).
        XCTAssertLessThan(r, 2.5)
        XCTAssertLessThan(r, baseline / 2)
    }

    /// A model trained WITHOUT NaN must behave identically well after the
    /// missing-bin layout change (regression guard).
    func testNoMissingUnaffected() throws {
        let (X, y) = DataGen.friedman(rows: 20_000, cols: 8, seed: 23)
        let b = try MacBooster(params: {
            var p = BoosterParams(); p.numTrees = 50; p.maxDepth = 5; return p
        }())
        try b.fit(featureMajor: X, rows: 20_000, cols: 8, labels: y)
        let (Xv, yv) = DataGen.friedman(rows: 5_000, cols: 8, seed: 24)
        // LightGBM scores 1.284 valid RMSE with this config.
        XCTAssertLessThan(rmse(b.predict(featureMajor: Xv, rows: 5_000, cols: 8), yv), 1.40)
    }

    // MARK: categorical (LightGBM test_engine.py::test_categorical_handle)

    /// y depends on membership in {0, 3} — a set no single ordered threshold
    /// can separate. A SINGLE depth-1 categorical tree must fit it
    /// near-perfectly; the same data as numeric cannot with one stump.
    /// (Boosted numeric stumps eventually can — additive step functions —
    /// so the contrast only holds at one tree.)
    func testCategoricalSubsetSplit() throws {
        let rows = 4_000
        var rng = SplitMix64(seed: 25)
        var X = [Float](repeating: 0, count: rows)
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows {
            let cat = Float(rng.next() % 4)
            X[i] = cat
            y[i] = (cat == 0 || cat == 3) ? 10 : 0
        }
        func run(categorical: Bool) throws -> Float {
            let b = try MacBooster(params: {
                var p = BoosterParams()
                p.numTrees = 1; p.maxDepth = 1; p.learningRate = 1.0; p.lambda = 0
                p.catSmooth = 0.01
                if categorical { p.categoricalFeatures = [0] }
                return p
            }())
            try b.fit(featureMajor: X, rows: rows, cols: 1, labels: y)
            return rmse(b.predict(featureMajor: X, rows: rows, cols: 1), y)
        }
        let catRMSE = try run(categorical: true)
        let numRMSE = try run(categorical: false)
        XCTAssertLessThan(catRMSE, 0.05, "subset split must separate {0,3} vs {1,2} exactly")
        XCTAssertGreaterThan(numRMSE, 1.0, "one ordered stump cannot separate {0,3}")
    }

    /// Categorical columns can contain NaN (missing joins the sorted order
    /// as one more category).
    func testCategoricalWithMissing() throws {
        let rows = 6_000
        var rng = SplitMix64(seed: 26)
        var X = [Float](repeating: 0, count: rows)
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows {
            let r = rng.next() % 5
            if r == 4 { X[i] = .nan; y[i] = 5 }
            else { X[i] = Float(r); y[i] = (r == 1 || r == 2) ? 10 : 0 }
        }
        let b = try MacBooster(params: {
            var p = BoosterParams()
            p.numTrees = 40; p.maxDepth = 2; p.learningRate = 0.3
            p.categoricalFeatures = [0]
            return p
        }())
        try b.fit(featureMajor: X, rows: rows, cols: 1, labels: y)
        let pred = b.predict(featureMajor: [0, 1, .nan], rows: 3, cols: 1)
        XCTAssertEqual(pred[0], 0, accuracy: 0.3)
        XCTAssertEqual(pred[1], 10, accuracy: 0.3)
        XCTAssertEqual(pred[2], 5, accuracy: 0.3)
    }

    /// Category ids outside 0..<numBins-1 must be rejected with a typed error.
    func testCategoricalCardinalityValidation() throws {
        let X: [Float] = [0, 1, 2, 999]
        let y: [Float] = [0, 1, 2, 3]
        let b = try MacBooster(params: {
            var p = BoosterParams(); p.numTrees = 1; p.categoricalFeatures = [0]; return p
        }())
        XCTAssertThrowsError(try b.fit(featureMajor: X, rows: 4, cols: 1, labels: y)) { err in
            guard case MacBoostError.categoricalOutOfRange = err else {
                return XCTFail("expected categoricalOutOfRange, got \(err)")
            }
        }
    }

    // MARK: early stopping (LightGBM test_engine.py::test_early_stopping)

    /// Small noisy data + aggressive learning rate overfits; early stopping
    /// must fire before numTrees, truncate to the best iteration, and the
    /// truncated model must be at least as good on the valid set as the
    /// overfit full model.
    func testEarlyStopping() throws {
        let (X, y) = DataGen.friedman(rows: 2_000, cols: 10, seed: 27)
        let (Xv, yv) = DataGen.friedman(rows: 2_000, cols: 10, seed: 28)
        let valid = EvalSet(featureMajor: Xv, rows: 2_000, labels: yv)

        func run(esRounds: Int) throws -> MacBooster {
            let b = try MacBooster(params: {
                var p = BoosterParams()
                p.numTrees = 400; p.maxDepth = 6; p.learningRate = 0.5
                return p
            }())
            try b.fit(featureMajor: X, rows: 2_000, cols: 10, labels: y,
                      valid: valid, earlyStoppingRounds: esRounds)
            return b
        }
        let es = try run(esRounds: 20)
        XCTAssertNotNil(es.bestIteration)
        XCTAssertLessThan(es.bestIteration!, 400, "lr=0.5 on 2k rows must overfit before 400 trees")
        XCTAssertEqual(es.trees.count, es.bestIteration!, "model must be truncated to best iteration")

        let full = try run(esRounds: 0)
        XCTAssertEqual(full.trees.count, 400)
        let esRMSE = rmse(es.predict(featureMajor: Xv, rows: 2_000, cols: 10), yv)
        let fullRMSE = rmse(full.predict(featureMajor: Xv, rows: 2_000, cols: 10), yv)
        XCTAssertLessThanOrEqual(esRMSE, fullRMSE + 0.02,
                                 "truncated-at-best model must not be worse than the overfit one")
    }

    /// evalHistory must be consistent with a from-scratch CPU predict of the
    /// truncated model (GPU per-tree valid eval == CPU raw-threshold eval).
    func testValidationHistoryConsistency() throws {
        let (X, y) = DataGen.friedman(rows: 10_000, cols: 8, seed: 29)
        let (Xv, yv) = DataGen.friedman(rows: 4_000, cols: 8, seed: 30)
        let b = try MacBooster(params: {
            var p = BoosterParams(); p.numTrees = 60; p.maxDepth = 5; return p
        }())
        try b.fit(featureMajor: X, rows: 10_000, cols: 8, labels: y,
                  valid: EvalSet(featureMajor: Xv, rows: 4_000, labels: yv),
                  earlyStoppingRounds: 30)
        guard let best = b.bestIteration else { return XCTFail("bestIteration unset") }
        XCTAssertEqual(b.trees.count, best)
        let recomputed = rmse(b.predict(featureMajor: Xv, rows: 4_000, cols: 8), yv)
        XCTAssertEqual(b.evalHistory[best - 1], recomputed, accuracy: 2e-3)
    }

    // MARK: GOSS (LightGBM test_engine.py::test_sample_strategy_with_boosting)

    /// GOSS trains on ~28% of rows per tree; accuracy must stay close to
    /// full-data training and far below the baseline.
    func testGOSSAccuracy() throws {
        let (X, y) = DataGen.friedman(rows: 40_000, cols: 10, seed: 41)
        let (Xv, yv) = DataGen.friedman(rows: 8_000, cols: 10, seed: 42)
        func run(goss: Bool) throws -> Float {
            let b = try MacBooster(params: {
                var p = BoosterParams()
                p.numTrees = 150; p.maxDepth = 6
                p.goss = goss
                return p
            }())
            try b.fit(featureMajor: X, rows: 40_000, cols: 10, labels: y)
            return rmse(b.predict(featureMajor: Xv, rows: 8_000, cols: 10), yv)
        }
        let full = try run(goss: false)
        let goss = try run(goss: true)
        XCTAssertLessThan(goss, full + 0.12,
                          "GOSS (\(goss)) must stay close to full training (\(full))")
        XCTAssertLessThan(goss, 1.35)
    }

    /// Invalid GOSS rates must be rejected.
    func testGOSSRateValidation() throws {
        var p = BoosterParams()
        p.goss = true; p.gossTopRate = 0.8; p.gossOtherRate = 0.5
        XCTAssertThrowsError(try MacBooster(params: p)) { err in
            guard case MacBoostError.invalidInput = err else {
                return XCTFail("expected invalidInput, got \(err)")
            }
        }
    }

    // MARK: binned dataset (.mbds, LightGBM save_binary pattern)

    /// makeDataset -> save -> load must round-trip exactly, and training
    /// from the dataset must match training from raw data (same binned
    /// matrix; only GPU-atomic noise differs).
    func testBinnedDatasetRoundTrip() throws {
        let rows = 20_000, cols = 9
        var (X, y) = DataGen.friedman(rows: rows, cols: 8, seed: 71)
        var rng = SplitMix64(seed: 72)
        for i in 0..<rows where rng.uniform() < 0.1 { X[i] = .nan }
        X.append(contentsOf: (0..<rows).map { _ in Float(rng.next() % 5) })
        for i in 0..<rows { y[i] += X[8 * rows + i] == 2 ? 3 : 0 }

        var p = BoosterParams()
        p.numTrees = 60; p.maxDepth = 5; p.categoricalFeatures = [8]
        let maker = try MacBooster(params: p)
        let ds = try maker.makeDataset(featureMajor: X, rows: rows, cols: cols,
                                       labels: y, featureNames: (0..<cols).map { "c\($0)" })

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("macboost-ds-\(UUID().uuidString).mbds")
        defer { try? FileManager.default.removeItem(at: url) }
        try ds.save(to: url)
        let loaded = try BinnedDataset.load(from: url)
        XCTAssertEqual(loaded.rows, ds.rows)
        XCTAssertEqual(loaded.numBins, ds.numBins)
        XCTAssertEqual(loaded.categorical, ds.categorical)
        XCTAssertEqual(loaded.featureNames, ds.featureNames)
        XCTAssertEqual(loaded.labels, ds.labels)
        XCTAssertEqual(loaded.bins, ds.bins)
        XCTAssertEqual(loaded.edges, ds.edges)

        // Note: fresh booster with NO categorical params — the dataset's
        // layout must win.
        var p2 = BoosterParams(); p2.numTrees = 60; p2.maxDepth = 5
        let fromDS = try MacBooster(params: p2)
        try fromDS.fit(dataset: loaded)
        let fromRaw = try MacBooster(params: p)
        try fromRaw.fit(featureMajor: X, rows: rows, cols: cols, labels: y)
        let rA = rmse(fromDS.predict(featureMajor: X, rows: rows, cols: cols), y)
        let rB = rmse(fromRaw.predict(featureMajor: X, rows: rows, cols: cols), y)
        XCTAssertEqual(rA, rB, accuracy: 0.03,
                       "training from .mbds must match training from raw data")
        XCTAssertEqual(fromDS.featureNames, ds.featureNames)
    }

    // MARK: GPU inference

    /// The GPU forest walk must be bit-identical to the CPU traversal,
    /// covering numeric, missing and categorical routing.
    func testGPUPredictMatchesCPUExactly() throws {
        let rows = 20_000, cols = 7
        var (X, y) = DataGen.friedman(rows: rows, cols: 6, seed: 73)
        var rng = SplitMix64(seed: 74)
        for i in 0..<rows where rng.uniform() < 0.15 { X[rows + i] = .nan }
        X.append(contentsOf: (0..<rows).map { _ in Float(rng.next() % 6) })
        for i in 0..<rows { y[i] += (X[6 * rows + i] == 1 || X[6 * rows + i] == 4) ? 4 : 0 }

        let b = try MacBooster(params: {
            var p = BoosterParams(); p.numTrees = 50; p.maxDepth = 6
            p.categoricalFeatures = [6]; return p
        }())
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y)

        let gpu = b.predict(featureMajor: X, rows: rows, cols: cols)   // >= 4096 -> GPU
        let cpu = b.predictCPU(featureMajor: X, rows: rows, cols: cols)
        XCTAssertEqual(gpu, cpu, "GPU forest inference must be bit-identical to CPU")
    }

    // MARK: serialization

    /// Save -> load -> predictions must be bitwise identical, covering
    /// numeric, missing and categorical routing in one model.
    func testSaveLoadRoundTrip() throws {
        let rows = 8_000, cols = 6
        var (X, y) = DataGen.friedman(rows: rows, cols: cols, seed: 31)
        var rng = SplitMix64(seed: 32)
        for i in 0..<rows {                       // col 5 becomes categorical
            let cat = Float(rng.next() % 6)
            X[5 * rows + i] = cat
            y[i] += (cat == 2 || cat == 4) ? 3 : 0
        }
        for i in 0..<rows where rng.uniform() < 0.1 { X[0 * rows + i] = .nan }

        let b = try MacBooster(params: {
            var p = BoosterParams()
            p.numTrees = 40; p.maxDepth = 5; p.categoricalFeatures = [5]
            return p
        }())
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y)

        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("macboost-roundtrip-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try b.save(to: url)
        let loaded = try MacBooster.load(from: url)

        XCTAssertEqual(loaded.numFeatures, b.numFeatures)
        XCTAssertEqual(loaded.baseScore, b.baseScore)
        XCTAssertEqual(loaded.trees.count, b.trees.count)

        var (Xv, _) = DataGen.friedman(rows: 2_000, cols: cols, seed: 33)
        for i in 0..<2_000 { Xv[5 * 2_000 + i] = Float(rng.next() % 6) }
        for i in 0..<2_000 where rng.uniform() < 0.1 { Xv[i] = .nan }
        let p1 = b.predict(featureMajor: Xv, rows: 2_000, cols: cols)
        let p2 = loaded.predict(featureMajor: Xv, rows: 2_000, cols: cols)
        XCTAssertEqual(p1, p2, "round-tripped model must predict bitwise identically")
    }
}
