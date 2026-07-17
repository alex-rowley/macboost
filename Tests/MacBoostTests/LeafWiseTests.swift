import XCTest
@testable import MacBoost

/// Leaf-wise (best-first) growth: num_leaves semantics, LightGBM-style.
final class LeafWiseTests: XCTestCase {

    private func rmse(_ pred: [Float], _ y: [Float]) -> Float {
        var s: Float = 0
        for i in 0..<y.count { let d = pred[i] - y[i]; s += d * d }
        return (s / Float(y.count)).squareRoot()
    }

    /// Heterogeneous target: a sharp interaction confined to a small
    /// region, linear elsewhere. Level-wise spends depth everywhere;
    /// best-first concentrates leaves where the signal is — at an equal
    /// leaf budget it must win.
    func testBeatsLevelWiseOnHeterogeneousData() throws {
        let rows = 20_000, cols = 6
        var rng = SplitMix64(seed: 501)
        var X = [Float](repeating: 0, count: rows * cols)
        for i in 0..<X.count { X[i] = rng.uniform() }
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows {
            let x0 = X[i], x1 = X[rows + i], x2 = X[2 * rows + i]
            y[i] = 2 * x1
            if x0 > 0.9 {   // deep structure in 10% of the space
                y[i] += 10 * ((x1 > 0.5) != (x2 > 0.5) ? 1 : -1)
                    * (x2 > 0.5 ? 1.5 : 1.0)
            }
            y[i] += 0.1 * rng.uniform()
        }

        var pLevel = BoosterParams()
        pLevel.numTrees = 60; pLevel.maxDepth = 4       // 16 leaves/tree
        let level = try MacBooster(params: pLevel)
        try level.fit(featureMajor: X, rows: rows, cols: cols, labels: y)

        var pLeaf = BoosterParams()
        pLeaf.numTrees = 60; pLeaf.maxDepth = 8; pLeaf.numLeaves = 16
        let leaf = try MacBooster(params: pLeaf)
        try leaf.fit(featureMajor: X, rows: rows, cols: cols, labels: y)

        let rLevel = rmse(level.predict(featureMajor: X, rows: rows, cols: cols), y)
        let rLeaf = rmse(leaf.predict(featureMajor: X, rows: rows, cols: cols), y)
        XCTAssertLessThan(rLeaf, rLevel,
            "equal leaf budget: best-first (\(rLeaf)) must beat level-wise (\(rLevel))")
    }

    /// On homogeneous data leaf-wise must at least match level-wise.
    func testMatchesLevelWiseOnFriedman() throws {
        let rows = 15_000
        let (X, y) = DataGen.friedman(rows: rows, cols: 8, seed: 511)
        var pLevel = BoosterParams(); pLevel.numTrees = 80
        let level = try MacBooster(params: pLevel)
        try level.fit(featureMajor: X, rows: rows, cols: 8, labels: y)
        var pLeaf = BoosterParams()
        pLeaf.numTrees = 80; pLeaf.maxDepth = 10; pLeaf.numLeaves = 64
        let leaf = try MacBooster(params: pLeaf)
        try leaf.fit(featureMajor: X, rows: rows, cols: 8, labels: y)
        let rLevel = rmse(level.predict(featureMajor: X, rows: rows, cols: 8), y)
        let rLeaf = rmse(leaf.predict(featureMajor: X, rows: rows, cols: 8), y)
        XCTAssertLessThan(rLeaf, rLevel * 1.05,
            "leaf-wise (\(rLeaf)) must be within 5% of level-wise (\(rLeaf)) on homogeneous data")
    }

    /// NaN + categorical routing, save/load round-trip parity, and the
    /// leaf-count budget actually being respected.
    func testMissingCategoricalAndRoundTrip() throws {
        let rows = 8_000, cols = 6
        var (X, y) = DataGen.friedman(rows: rows, cols: 5, seed: 521)
        var rng = SplitMix64(seed: 522)
        for i in 0..<rows where rng.uniform() < 0.2 { X[rows + i] = .nan }
        let cat = (0..<rows).map { _ in Float(rng.next() % 5) }
        X.append(contentsOf: cat)
        for i in 0..<rows { y[i] += cat[i] == 2 || cat[i] == 4 ? 4 : -4 }

        var p = BoosterParams()
        p.numTrees = 60; p.maxDepth = 10; p.numLeaves = 31
        p.categoricalFeatures = [5]
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y)
        let r = rmse(b.predict(featureMajor: X, rows: rows, cols: cols), y)
        XCTAssertLessThan(r, 1.6, "NaN+cat leaf-wise RMSE \(r)")

        for tree in b.trees {
            var leaves = 0
            var stack = [0]
            while let n = stack.popLast() {
                if tree.feature[n] >= 0 { stack += [2 * n + 1, 2 * n + 2] }
                else { leaves += 1 }
            }
            XCTAssertLessThanOrEqual(leaves, 31, "leaf budget exceeded (\(leaves))")
        }

        let tmp = FileManager.default.temporaryDirectory
            .appendingPathComponent("lw-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: tmp) }
        try b.save(to: tmp)
        let loaded = try MacBooster.load(from: tmp)
        XCTAssertEqual(b.predict(featureMajor: X, rows: rows, cols: cols),
                       loaded.predict(featureMajor: X, rows: rows, cols: cols),
                       "leaf-wise model must round-trip save/load bitwise")
    }

    /// Multiclass, GOSS, bagging and colsample all compose with leaf-wise.
    func testComposition() throws {
        let rows = 8_000
        var rng = SplitMix64(seed: 531)
        var X = [Float](repeating: 0, count: rows * 4)
        for i in 0..<X.count { X[i] = rng.uniform() }
        let ym = (0..<rows).map { i in Float(min(2, Int(X[i] * 3))) }
        var pm = BoosterParams()
        pm.numTrees = 30; pm.maxDepth = 8; pm.numLeaves = 15
        pm.objective = .multiclass; pm.numClasses = 3
        let mc = try MacBooster(params: pm)
        try mc.fit(featureMajor: X, rows: rows, cols: 4, labels: ym)
        let pred = mc.predict(featureMajor: X, rows: rows, cols: 4)
        var correct = 0
        for i in 0..<rows {
            var bestK = 0
            for k in 1..<3 where pred[i * 3 + k] > pred[i * 3 + bestK] { bestK = k }
            if Float(bestK) == ym[i] { correct += 1 }
        }
        XCTAssertGreaterThan(Float(correct) / Float(rows), 0.95,
                             "leaf-wise multiclass accuracy")

        let (Xg, yg) = DataGen.friedman(rows: rows, cols: 5, seed: 532)
        var pg = BoosterParams()
        pg.numTrees = 60; pg.maxDepth = 8; pg.numLeaves = 31; pg.goss = true
        let g = try MacBooster(params: pg)
        try g.fit(featureMajor: Xg, rows: rows, cols: 5, labels: yg)
        XCTAssertLessThan(rmse(g.predict(featureMajor: Xg, rows: rows, cols: 5), yg), 1.8,
                          "leaf-wise GOSS")

        var pb = BoosterParams()
        pb.numTrees = 60; pb.maxDepth = 8; pb.numLeaves = 31
        pb.subsample = 0.7; pb.featureFraction = 0.8
        let bg = try MacBooster(params: pb)
        try bg.fit(featureMajor: Xg, rows: rows, cols: 5, labels: yg)
        XCTAssertLessThan(rmse(bg.predict(featureMajor: Xg, rows: rows, cols: 5), yg), 1.8,
                          "leaf-wise bagging + colsample")
    }

    /// Monotone constraints hold under best-first growth.
    func testMonotoneConstraint() throws {
        let rows = 8_000
        var rng = SplitMix64(seed: 541)
        var X = [Float](repeating: 0, count: rows * 3)
        for i in 0..<X.count { X[i] = rng.uniform() }
        let y = (0..<rows).map { i in
            3 * X[i] + X[rows + i] + 0.3 * rng.uniform()
        }
        var p = BoosterParams()
        p.numTrees = 60; p.maxDepth = 8; p.numLeaves = 31
        p.monotoneConstraints = [1, 0, 0]
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: 3, labels: y)

        // Sweep feature 0 with the others fixed; prediction must not fall.
        let steps = 40
        var probe = [Float](repeating: 0.5, count: steps * 3)
        for s in 0..<steps { probe[s] = Float(s) / Float(steps - 1) }
        let out = b.predict(featureMajor: probe, rows: steps, cols: 3)
        for s in 1..<steps {
            XCTAssertGreaterThanOrEqual(out[s], out[s - 1] - 1e-5,
                "monotone violated at step \(s): \(out[s - 1]) -> \(out[s])")
        }
    }

    /// Quantile objective with leaf renewal under leaf-wise growth.
    func testQuantileRenewal() throws {
        let rows = 8_000
        let (X, y) = DataGen.friedman(rows: rows, cols: 5, seed: 551)
        var p = BoosterParams()
        p.numTrees = 80; p.maxDepth = 8; p.numLeaves = 31
        p.objective = .quantile; p.alpha = 0.9
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: 5, labels: y)
        let pred = b.predict(featureMajor: X, rows: rows, cols: 5)
        var covered = 0
        for i in 0..<rows where y[i] <= pred[i] { covered += 1 }
        let cov = Float(covered) / Float(rows)
        XCTAssertEqual(cov, 0.9, accuracy: 0.05,
                       "alpha=0.9 coverage \(cov) (leaf renewal under leaf-wise)")
    }

    /// Early stopping through the shared metric tail.
    func testEarlyStoppingWithValid() throws {
        let (X, y) = DataGen.friedman(rows: 12_000, cols: 6, seed: 561)
        let (Xv, yv) = DataGen.friedman(rows: 3_000, cols: 6, seed: 562)
        var p = BoosterParams()
        p.numTrees = 400; p.maxDepth = 8; p.numLeaves = 31
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: 12_000, cols: 6, labels: y,
                  valid: EvalSet(featureMajor: Xv, rows: 3_000, labels: yv),
                  earlyStoppingRounds: 20)
        XCTAssertNotNil(b.bestIteration, "early stopping must engage")
        XCTAssertLessThan(rmse(b.predict(featureMajor: Xv, rows: 3_000, cols: 6), yv),
                          1.3, "valid RMSE at noise floor")
    }

    func testValidation() throws {
        var p = BoosterParams()
        p.maxDepth = 4; p.numLeaves = 17    // > 2^4
        XCTAssertThrowsError(try MacBooster(params: p),
                             "numLeaves > 2^maxDepth must throw")
        var p2 = BoosterParams()
        p2.maxDepth = 4; p2.numLeaves = 1
        XCTAssertThrowsError(try MacBooster(params: p2), "numLeaves < 2 must throw")

        // numLeaves = 2: every tree is a stump.
        let (X, y) = DataGen.friedman(rows: 2_000, cols: 5, seed: 571)
        var p3 = BoosterParams()
        p3.numTrees = 30; p3.numLeaves = 2
        let b = try MacBooster(params: p3)
        try b.fit(featureMajor: X, rows: 2_000, cols: 5, labels: y)
        for tree in b.trees {
            XCTAssertTrue(tree.feature[0] >= 0 || b.trees.count == 30)
            XCTAssertLessThan(tree.feature.dropFirst(3).filter { $0 >= 0 }.count, 1,
                              "numLeaves=2 must produce stumps")
        }
    }
}
