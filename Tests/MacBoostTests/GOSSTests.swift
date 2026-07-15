import XCTest
@testable import MacBoost

/// GOSS is a supported training mode, so the standard behavioural
/// guarantees are asserted under it explicitly — not just a single
/// accuracy comparison. Adapted from LightGBM's data_sample_strategy=goss
/// coverage in test_engine.py.
final class GOSSTests: XCTestCase {

    func rmse(_ a: [Float], _ b: [Float]) -> Float {
        var s = 0.0
        for i in 0..<a.count { let d = Double(a[i] - b[i]); s += d * d }
        return Float((s / Double(a.count)).squareRoot())
    }

    func gossParams(_ configure: (inout BoosterParams) -> Void = { _ in }) -> BoosterParams {
        var p = BoosterParams()
        p.goss = true
        configure(&p)
        return p
    }

    /// With topRate ~1 everything is kept at weight 1, so GOSS must behave
    /// like full training (within GPU-atomic run-to-run noise).
    func testDegeneratesToFullTrainingAtHighTopRate() throws {
        let (X, y) = DataGen.friedman(rows: 20_000, cols: 8, seed: 51)
        let (Xv, yv) = DataGen.friedman(rows: 5_000, cols: 8, seed: 52)
        func run(_ p: BoosterParams) throws -> Float {
            let b = try MacBooster(params: p)
            try b.fit(featureMajor: X, rows: 20_000, cols: 8, labels: y)
            return rmse(b.predict(featureMajor: Xv, rows: 5_000, cols: 8), yv)
        }
        var full = BoosterParams(); full.numTrees = 60; full.maxDepth = 5
        let fullRMSE = try run(full)
        let gossRMSE = try run(gossParams { p in
            p.numTrees = 60; p.maxDepth = 5
            p.gossTopRate = 0.95; p.gossOtherRate = 0.05
        })
        XCTAssertEqual(gossRMSE, fullRMSE, accuracy: 0.03)
    }

    /// The first ceil(1/learningRate) trees train on full data (LightGBM's
    /// warm-up): a model that stops inside the warm-up must match full
    /// training even with aggressive sampling rates configured.
    func testWarmupTreesUseFullData() throws {
        let (X, y) = DataGen.friedman(rows: 20_000, cols: 8, seed: 53)
        let (Xv, yv) = DataGen.friedman(rows: 5_000, cols: 8, seed: 54)
        func run(goss: Bool) throws -> Float {
            var p = BoosterParams()
            p.numTrees = 2; p.maxDepth = 5; p.learningRate = 0.5   // warm-up = 2
            if goss { p.goss = true; p.gossTopRate = 0.05; p.gossOtherRate = 0.05 }
            let b = try MacBooster(params: p)
            try b.fit(featureMajor: X, rows: 20_000, cols: 8, labels: y)
            return rmse(b.predict(featureMajor: Xv, rows: 5_000, cols: 8), yv)
        }
        let full = try run(goss: false)
        let goss = try run(goss: true)
        XCTAssertEqual(goss, full, accuracy: 0.02,
                       "inside the warm-up GOSS must be identical to full training")
    }

    /// Sampling must actually engage after warm-up: training on ~2% of rows
    /// has to be measurably worse than full data at the same tree count.
    func testSamplingActuallyEngages() throws {
        let (X, y) = DataGen.friedman(rows: 30_000, cols: 8, seed: 55)
        let (Xv, yv) = DataGen.friedman(rows: 5_000, cols: 8, seed: 56)
        func run(_ p: BoosterParams) throws -> Float {
            let b = try MacBooster(params: p)
            try b.fit(featureMajor: X, rows: 30_000, cols: 8, labels: y)
            return rmse(b.predict(featureMajor: Xv, rows: 5_000, cols: 8), yv)
        }
        var full = BoosterParams(); full.numTrees = 80; full.maxDepth = 5
        let fullRMSE = try run(full)
        let tinyRMSE = try run(gossParams { p in
            p.numTrees = 80; p.maxDepth = 5
            p.gossTopRate = 0.01; p.gossOtherRate = 0.01
        })
        XCTAssertGreaterThan(tinyRMSE, fullRMSE + 0.03,
                             "2% sampling must visibly degrade vs full data — " +
                             "otherwise GOSS is silently not sampling")
    }

    /// Standard-suite invariant under GOSS: binary logistic reaches the
    /// same Bayes-floor region as full training (see testBinaryLogistic).
    func testBinaryLogisticUnderGOSS() throws {
        let rows = 20_000, cols = 4
        var rng = SplitMix64(seed: 57)
        var X = [Float](repeating: 0, count: rows * cols)
        for k in 0..<X.count { X[k] = rng.uniform() }
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows {
            y[i] = (X[i] + X[rows + i] + 0.3 * rng.normal()) > 1 ? 1 : 0
        }
        let b = try MacBooster(params: gossParams { p in
            p.numTrees = 60; p.maxDepth = 5; p.objective = .binaryLogistic
        })
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y)
        let raw = b.predict(featureMajor: X, rows: rows, cols: cols)
        var ll = 0.0
        for i in 0..<rows {
            let p = 1 / (1 + exp(-Double(raw[i])))
            ll += -(Double(y[i]) * log(max(p, 1e-12))
                    + (1 - Double(y[i])) * log(max(1 - p, 1e-12)))
        }
        XCTAssertLessThan(Float(ll / Double(rows)), 0.44,
                          "GOSS logloss must stay near the 0.407 Bayes floor")
    }

    /// Standard-suite invariant under GOSS: missing values and categorical
    /// features keep working (sampled histograms + subset splits + NaN bin).
    func testMissingAndCategoricalUnderGOSS() throws {
        let rows = 30_000, cols = 9
        var (X, y) = DataGen.friedman(rows: rows, cols: 8, seed: 58)
        var rng = SplitMix64(seed: 59)
        for i in 0..<rows where rng.uniform() < 0.2 { X[i] = .nan }
        X.append(contentsOf: (0..<rows).map { _ in Float(rng.next() % 6) })
        for i in 0..<rows {
            let c = X[8 * rows + i]
            y[i] += (c == 1 || c == 4) ? 4 : -4
        }
        let b = try MacBooster(params: gossParams { p in
            p.numTrees = 120; p.maxDepth = 6; p.categoricalFeatures = [8]
        })
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y)
        let r = rmse(b.predict(featureMajor: X, rows: rows, cols: cols), y)
        let baseline = rmse(y, [Float](repeating: b.baseScore, count: rows))
        XCTAssertLessThan(r, baseline / 3,
                          "GOSS + NaN + categorical must still learn the categorical signal")
        XCTAssertLessThan(r, 2.2)
    }

    /// Standard-suite invariant under GOSS: early stopping fires, truncates
    /// to the best iteration, and the GPU validation metric matches a CPU
    /// recompute of the truncated model.
    func testEarlyStoppingUnderGOSS() throws {
        let (X, y) = DataGen.friedman(rows: 3_000, cols: 10, seed: 60)
        let (Xv, yv) = DataGen.friedman(rows: 3_000, cols: 10, seed: 61)
        let b = try MacBooster(params: gossParams { p in
            p.numTrees = 400; p.maxDepth = 6; p.learningRate = 0.5
        })
        try b.fit(featureMajor: X, rows: 3_000, cols: 10, labels: y,
                  valid: EvalSet(featureMajor: Xv, rows: 3_000, labels: yv),
                  earlyStoppingRounds: 20)
        guard let best = b.bestIteration else { return XCTFail("bestIteration unset") }
        XCTAssertLessThan(best, 400, "aggressive lr on 3k rows must overfit and stop early")
        XCTAssertEqual(b.trees.count, best)
        let recomputed = rmse(b.predict(featureMajor: Xv, rows: 3_000, cols: 10), yv)
        XCTAssertEqual(b.evalHistory[best - 1], recomputed, accuracy: 2e-3,
                       "GPU valid eval must match CPU predict under GOSS")
    }

    /// Standard-suite invariant under GOSS: save -> load is bitwise exact.
    func testSaveLoadRoundTripUnderGOSS() throws {
        let (X, y) = DataGen.friedman(rows: 10_000, cols: 6, seed: 62)
        let b = try MacBooster(params: gossParams { p in
            p.numTrees = 40; p.maxDepth = 5
        })
        try b.fit(featureMajor: X, rows: 10_000, cols: 6, labels: y)
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("macboost-goss-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try b.save(to: url)
        let loaded = try MacBooster.load(from: url)
        let (Xv, _) = DataGen.friedman(rows: 2_000, cols: 6, seed: 63)
        XCTAssertEqual(b.predict(featureMajor: Xv, rows: 2_000, cols: 6),
                       loaded.predict(featureMajor: Xv, rows: 2_000, cols: 6))
    }
}
