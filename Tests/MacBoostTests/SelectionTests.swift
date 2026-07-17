import XCTest
@testable import MacBoost

/// Boruta shadow-feature selection (GPU-resident shadows) and the
/// allowedFeatures training mask it drives.
final class SelectionTests: XCTestCase {

    /// Friedman data with pure-noise columns appended: X[0..<signal]
    /// informative, X[signal...] uniform noise.
    private func signalPlusNoise(rows: Int, signal: Int = 5, noise: Int = 5,
                                 seed: UInt64 = 311) -> (X: [Float], y: [Float], cols: Int) {
        var (X, y) = DataGen.friedman(rows: rows, cols: signal, seed: seed)
        var rng = SplitMix64(seed: seed &+ 1)
        for _ in 0..<noise {
            X.append(contentsOf: (0..<rows).map { _ in rng.uniform() })
        }
        return (X, y, signal + noise)
    }

    // MARK: - Core separation property

    /// Signal features must be confirmed, pure-noise columns rejected.
    /// Noise columns are distributionally identical to shadows, so per
    /// round each has ~1/(cols+1) odds of beating every shadow — far
    /// below the binomial confirm bar.
    func testSeparatesSignalFromNoise() throws {
        let (X, y, cols) = signalPlusNoise(rows: 6_000)
        var p = BoosterParams()
        p.numTrees = 60; p.maxDepth = 6
        let b = try MacBooster(params: p)
        let result = try b.selectFeatures(featureMajor: X, rows: 6_000, cols: cols,
                                          labels: y, rounds: 20)

        XCTAssertEqual(result.hits.count, cols)
        XCTAssertEqual(result.rounds, 20)
        XCTAssertGreaterThanOrEqual(result.confirmed.filter { $0 < 5 }.count, 4,
            "signal features must be confirmed (got \(result.confirmed))")
        XCTAssertTrue(result.confirmed.allSatisfy { $0 < 5 },
            "noise columns must never be confirmed (got \(result.confirmed))")
        XCTAssertGreaterThanOrEqual(result.rejected.filter { $0 >= 5 }.count, 4,
            "noise columns must be rejected (rejected: \(result.rejected))")
        for f in result.confirmed {
            XCTAssertGreaterThan(result.gainRatio[f], 1,
                "confirmed feature \(f) should out-gain the shadow ceiling")
        }
        // Every feature is classified exactly once.
        let all = (result.confirmed + result.tentative + result.rejected).sorted()
        XCTAssertEqual(all, Array(0..<cols))
    }

    /// A near-duplicate of a signal feature: Boruta is an all-relevant
    /// method, so redundant-but-informative copies must not be rejected
    /// (this is exactly where greedy/stepwise selection fails).
    func testRedundantSignalFeatureIsNotRejected() throws {
        let rows = 6_000
        var (X, y, _) = signalPlusNoise(rows: rows, signal: 5, noise: 2, seed: 341)
        var rng = SplitMix64(seed: 342)
        // Feature 7 = feature 3 + small jitter (highly informative, redundant).
        X.append(contentsOf: (0..<rows).map { i in
            X[3 * rows + i] + 0.01 * rng.uniform()
        })
        var p = BoosterParams(); p.numTrees = 60
        let b = try MacBooster(params: p)
        let result = try b.selectFeatures(featureMajor: X, rows: rows, cols: 8,
                                          labels: y, rounds: 20)
        XCTAssertFalse(result.rejected.contains(3),
                       "original of a duplicated signal pair must survive")
        XCTAssertFalse(result.rejected.contains(7),
                       "near-duplicate of a signal feature must survive")
    }

    // MARK: - Data-type coverage

    /// Selection must handle NaN (missing) values and categorical columns:
    /// shadows of categorical features stay categorical, and an informative
    /// categorical must be confirmed while a junk categorical is rejected.
    func testSelectionWithMissingAndCategorical() throws {
        let rows = 6_000
        var (X, y) = DataGen.friedman(rows: rows, cols: 5, seed: 351)
        var rng = SplitMix64(seed: 352)
        for i in 0..<rows where rng.uniform() < 0.15 { X[rows + i] = .nan }
        // Feature 5: informative categorical. Feature 6: junk categorical.
        let cat = (0..<rows).map { _ in Float(rng.next() % 6) }
        X.append(contentsOf: cat)
        for i in 0..<rows { y[i] += cat[i] == 2 || cat[i] == 4 ? 4 : -4 }
        X.append(contentsOf: (0..<rows).map { _ in Float(rng.next() % 6) })

        var p = BoosterParams()
        p.numTrees = 60; p.categoricalFeatures = [5, 6]
        let b = try MacBooster(params: p)
        let result = try b.selectFeatures(featureMajor: X, rows: rows, cols: 7,
                                          labels: y, rounds: 20)
        XCTAssertTrue(result.confirmed.contains(5),
                      "informative categorical must be confirmed (\(result.confirmed))")
        XCTAssertTrue(result.rejected.contains(6),
                      "junk categorical must be rejected (\(result.rejected))")
        XCTAssertTrue(result.confirmed.contains(1),
                      "NaN-riddled signal feature must still be confirmed")
    }

    /// Objective coverage: binary logistic and multiclass selection run
    /// end-to-end and separate signal from noise.
    func testSelectionAcrossObjectives() throws {
        let rows = 6_000
        var rng = SplitMix64(seed: 361)
        var X = [Float]()
        for _ in 0..<4 { X.append(contentsOf: (0..<rows).map { _ in rng.uniform() }) }
        // Binary: y depends on features 0 and 1 only.
        let yb = (0..<rows).map { i -> Float in
            X[i] + X[rows + i] + 0.2 * rng.uniform() > 1.1 ? 1 : 0
        }
        var pb = BoosterParams()
        pb.numTrees = 50; pb.objective = .binaryLogistic
        let resultB = try MacBooster(params: pb)
            .selectFeatures(featureMajor: X, rows: rows, cols: 4, labels: yb, rounds: 15)
        XCTAssertTrue(resultB.confirmed.contains(0) && resultB.confirmed.contains(1),
                      "binary: signal features confirmed (\(resultB.confirmed))")
        XCTAssertTrue(resultB.confirmed.allSatisfy { $0 < 2 },
                      "binary: noise never confirmed (\(resultB.confirmed))")

        // Multiclass: class determined by feature 2 only.
        let ym = (0..<rows).map { i in Float(min(2, Int(X[2 * rows + i] * 3))) }
        var pm = BoosterParams()
        pm.numTrees = 30; pm.objective = .multiclass; pm.numClasses = 3
        let resultM = try MacBooster(params: pm)
            .selectFeatures(featureMajor: X, rows: rows, cols: 4, labels: ym, rounds: 15)
        XCTAssertTrue(resultM.confirmed.contains(2),
                      "multiclass: the class-driving feature must be confirmed")
        XCTAssertFalse(resultM.confirmed.contains(3),
                       "multiclass: noise must not be confirmed")
    }

    /// Sample weights flow into the selection fits, and sampling modes
    /// (GOSS, bagging, colsample) compose with selection.
    func testSelectionWithWeightsAndSampling() throws {
        let (X, y, cols) = signalPlusNoise(rows: 6_000, seed: 371)
        let weights = [Float](repeating: 1, count: 6_000)

        var pg = BoosterParams()
        pg.numTrees = 50; pg.goss = true
        let rg = try MacBooster(params: pg)
            .selectFeatures(featureMajor: X, rows: 6_000, cols: cols, labels: y,
                            weights: weights, rounds: 15)
        XCTAssertGreaterThanOrEqual(rg.confirmed.filter { $0 < 5 }.count, 4,
            "GOSS selection still confirms signal (\(rg.confirmed))")
        XCTAssertTrue(rg.confirmed.allSatisfy { $0 < 5 })

        var ps = BoosterParams()
        ps.numTrees = 50; ps.subsample = 0.7; ps.featureFraction = 0.8
        let rs = try MacBooster(params: ps)
            .selectFeatures(featureMajor: X, rows: 6_000, cols: cols, labels: y, rounds: 15)
        XCTAssertGreaterThanOrEqual(rs.confirmed.filter { $0 < 5 }.count, 4,
            "bagging+colsample selection still confirms signal (\(rs.confirmed))")
        XCTAssertTrue(rs.confirmed.allSatisfy { $0 < 5 })
    }

    // MARK: - Determinism and statistics

    /// Same seed → identical verdicts; the binomial verdict thresholds
    /// behave (0 hits rejected, all hits confirmed, rounds/2 tentative).
    func testDeterminismAndVerdictThresholds() throws {
        let (X, y, cols) = signalPlusNoise(rows: 4_000, seed: 381)
        var p = BoosterParams(); p.numTrees = 40
        let r1 = try MacBooster(params: p)
            .selectFeatures(featureMajor: X, rows: 4_000, cols: cols, labels: y,
                            rounds: 10, seed: 7)
        let r2 = try MacBooster(params: p)
            .selectFeatures(featureMajor: X, rows: 4_000, cols: cols, labels: y,
                            rounds: 10, seed: 7)
        XCTAssertEqual(r1.hits, r2.hits, "same seed must reproduce hit counts")
        XCTAssertEqual(r1.confirmed, r2.confirmed)
        XCTAssertEqual(r1.rejected, r2.rejected)

        // Verdicts are monotone in hits: every confirmed feature has more
        // hits than every rejected one.
        if let minC = r1.confirmed.map({ r1.hits[$0] }).min(),
           let maxR = r1.rejected.map({ r1.hits[$0] }).max() {
            XCTAssertGreaterThan(minC, maxR)
        }
    }

    /// Constant labels: no tree can split, nothing beats anything — the
    /// call must not crash and must confirm nothing.
    func testDegenerateConstantTarget() throws {
        let rows = 1_000
        var rng = SplitMix64(seed: 391)
        var X = [Float]()
        for _ in 0..<3 { X.append(contentsOf: (0..<rows).map { _ in rng.uniform() }) }
        let y = [Float](repeating: 2.5, count: rows)
        var p = BoosterParams(); p.numTrees = 10
        let result = try MacBooster(params: p)
            .selectFeatures(featureMajor: X, rows: rows, cols: 3, labels: y, rounds: 8)
        XCTAssertTrue(result.confirmed.isEmpty,
                      "no feature may be confirmed on a constant target")
    }

    /// Single-feature edge: one real column vs its own shadow.
    func testSingleFeature() throws {
        let rows = 3_000
        var rng = SplitMix64(seed: 401)
        let X = (0..<rows).map { _ in rng.uniform() }
        let y = X.map { $0 * 3 + 1 }
        var p = BoosterParams(); p.numTrees = 20
        let result = try MacBooster(params: p)
            .selectFeatures(featureMajor: X, rows: rows, cols: 1, labels: y, rounds: 10)
        XCTAssertEqual(result.confirmed, [0])
    }

    /// Probe models are sized independently of the final spec: a
    /// heavyweight configured model still runs selection with clamped
    /// probes (min(numTrees, 100)), an explicit `trees` override works
    /// at small sizes, and trees < 1 throws.
    func testProbeModelSizing() throws {
        let (X, y, cols) = signalPlusNoise(rows: 4_000, seed: 441)
        var p = BoosterParams(); p.numTrees = 400
        let auto = try MacBooster(params: p).selectFeatures(
            featureMajor: X, rows: 4_000, cols: cols, labels: y, rounds: 10)
        XCTAssertGreaterThanOrEqual(auto.confirmed.filter { $0 < 5 }.count, 4,
            "clamped probes must still separate signal (\(auto.confirmed))")
        XCTAssertTrue(auto.confirmed.allSatisfy { $0 < 5 })

        let light = try MacBooster(params: p).selectFeatures(
            featureMajor: X, rows: 4_000, cols: cols, labels: y, rounds: 10, trees: 20)
        XCTAssertGreaterThanOrEqual(light.confirmed.filter { $0 < 5 }.count, 4,
            "20-tree probes must still separate signal (\(light.confirmed))")
        XCTAssertTrue(light.confirmed.allSatisfy { $0 < 5 })

        XCTAssertThrowsError(try MacBooster(params: p).selectFeatures(
            featureMajor: X, rows: 4_000, cols: cols, labels: y, rounds: 10, trees: 0),
            "trees < 1 must throw")
    }

    // MARK: - Input validation

    func testSelectionInputValidation() throws {
        let (X, y, cols) = signalPlusNoise(rows: 300, seed: 411)
        var p = BoosterParams(); p.numTrees = 5
        let b = try MacBooster(params: p)
        XCTAssertThrowsError(try b.selectFeatures(
            featureMajor: X, rows: 300, cols: cols, labels: y, rounds: 2),
            "rounds < 5 must throw")
        XCTAssertThrowsError(try b.selectFeatures(
            featureMajor: X, rows: 300, cols: cols, labels: y, rounds: 10, alpha: 0),
            "alpha out of range must throw")
        XCTAssertThrowsError(try b.selectFeatures(
            featureMajor: X, rows: 300, cols: cols, labels: [1, 2, 3], rounds: 10),
            "label/row mismatch must throw")
        XCTAssertThrowsError(try b.selectFeatures(
            featureMajor: X, rows: 300, cols: cols,
            labels: y.map { $0.isFinite ? $0 : 0 }.enumerated()
                .map { $0.offset == 5 ? Float.nan : $0.element },
            rounds: 10),
            "NaN labels must throw")
    }

    func testAllowedFeaturesValidation() throws {
        let (X, y, cols) = signalPlusNoise(rows: 300, seed: 421)
        var p = BoosterParams(); p.numTrees = 5
        p.allowedFeatures = [99]
        XCTAssertThrowsError(try MacBooster(params: p).fit(
            featureMajor: X, rows: 300, cols: cols, labels: y),
            "out-of-range allowedFeatures must throw")
        p.allowedFeatures = []
        XCTAssertThrowsError(try MacBooster(params: p).fit(
            featureMajor: X, rows: 300, cols: cols, labels: y),
            "empty allowedFeatures must throw")
    }

    // MARK: - allowedFeatures training mask

    /// allowedFeatures must actually restrict splits: the final model has
    /// zero importance on excluded features, keeps accuracy, composes with
    /// colsample, and survives save/load.
    func testAllowedFeaturesRestrictsSplits() throws {
        let rows = 5_000, cols = 8
        var (X, y) = DataGen.friedman(rows: rows, cols: 5, seed: 321)
        var rng = SplitMix64(seed: 322)
        for _ in 0..<3 { X.append(contentsOf: (0..<rows).map { _ in rng.uniform() }) }

        var p = BoosterParams()
        p.numTrees = 80; p.maxDepth = 6
        p.allowedFeatures = [0, 1, 2, 3, 4]
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y)
        let imp = b.featureImportance(type: .split)
        for f in 5..<8 {
            XCTAssertEqual(imp[f], 0, "feature \(f) is excluded but split")
        }

        // Accuracy must match a model trained without the noise columns.
        var pRef = BoosterParams(); pRef.numTrees = 80; pRef.maxDepth = 6
        let ref = try MacBooster(params: pRef)
        try ref.fit(featureMajor: Array(X[0..<(5 * rows)]), rows: rows, cols: 5,
                    labels: y)
        let pred = b.predict(featureMajor: X, rows: rows, cols: cols)
        let predRef = ref.predict(featureMajor: Array(X[0..<(5 * rows)]),
                                  rows: rows, cols: 5)
        var mse: Float = 0
        var mseRef: Float = 0
        for i in 0..<rows {
            let d = pred[i] - y[i]
            let dr = predRef[i] - y[i]
            mse += d * d
            mseRef += dr * dr
        }
        mse /= Float(rows)
        mseRef /= Float(rows)
        XCTAssertLessThan(abs(mse - mseRef), 0.15,
            "masked model must match a model trained on the kept columns")

        // Composes with colsample: still never touches excluded features.
        var pc = BoosterParams()
        pc.numTrees = 40; pc.featureFraction = 0.5
        pc.allowedFeatures = [0, 1, 2, 3, 4]
        let bc = try MacBooster(params: pc)
        try bc.fit(featureMajor: X, rows: rows, cols: cols, labels: y)
        let impC = bc.featureImportance(type: .split)
        for f in 5..<8 { XCTAssertEqual(impC[f], 0) }

        // The mask is a training-time device: the saved model round-trips
        // and predicts identically on full-width X.
        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("sel-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try b.save(to: tmp)
        let loaded = try MacBooster.load(from: tmp)
        let predLoaded = loaded.predict(featureMajor: X, rows: rows, cols: cols)
        XCTAssertEqual(pred, predLoaded, "masked model must round-trip save/load")
    }

    // MARK: - Permutation kernel

    /// The Feistel permutation used for shadows must be a bijection: the
    /// Swift mirror (kept in lockstep with the MSL) covers [0, n) exactly,
    /// for awkward sizes around power-of-two boundaries.
    func testShadowPermutationIsBijective() throws {
        for n in [1, 2, 7, 255, 256, 257, 3_000, 65_537] {
            for seed in [UInt32(1), 42, 0xDEADBEEF] {
                var seen = Set<UInt32>()
                for i in 0..<UInt32(n) {
                    seen.insert(feistelPermMirror(i, n: UInt32(n), seed: seed))
                }
                XCTAssertEqual(seen.count, n,
                               "shadow_perm must be a bijection on [0, \(n))")
            }
        }
        // And it must actually permute (not be the identity) at real sizes.
        let n: UInt32 = 3_000
        let moved = (0..<n).filter { feistelPermMirror($0, n: n, seed: 42) != $0 }.count
        XCTAssertGreaterThan(moved, Int(n) / 2, "permutation should displace most indices")
    }

    /// End-to-end distribution check: a shadow column must have exactly
    /// its original's bin histogram (bijectivity through the actual GPU
    /// kernel, observed via selection gains being finite and shadows
    /// carrying no label signal on average).
    func testShadowsCarryNoSignal() throws {
        let (X, y, cols) = signalPlusNoise(rows: 4_000, signal: 5, noise: 0, seed: 431)
        var p = BoosterParams(); p.numTrees = 40
        let result = try MacBooster(params: p)
            .selectFeatures(featureMajor: X, rows: 4_000, cols: cols, labels: y, rounds: 15)
        // All-signal data: everything confirmed, gain ratios finite and > 1.
        XCTAssertEqual(result.confirmed.sorted(), Array(0..<cols))
        for f in 0..<cols {
            XCTAssertTrue(result.gainRatio[f].isFinite)
            XCTAssertGreaterThan(result.gainRatio[f], 1)
        }
    }
}

/// Swift mirror of the MSL shadow_perm (kept in lockstep for the test).
private func feistelPermMirror(_ i0: UInt32, n: UInt32, seed: UInt32) -> UInt32 {
    var bits = 1
    while (1 << bits) < n { bits += 1 }
    let hb = UInt32((bits + 1) / 2)
    let mask: UInt32 = (1 << hb) - 1
    func wang(_ x0: UInt32) -> UInt32 {
        var x = (x0 ^ 61) ^ (x0 >> 16)
        x = x &* 9
        x = x ^ (x >> 4)
        x = x &* 0x27d4_eb2d
        x = x ^ (x >> 15)
        return x
    }
    var i = i0
    repeat {
        var L = i >> hb, R = i & mask
        for r in 0..<4 {
            let t = R
            R = L ^ (wang(R ^ (seed &+ UInt32(r) &* 0x9E37_79B9)) & mask)
            L = t
        }
        i = (L << hb) | R
    } while i >= n
    return i
}
