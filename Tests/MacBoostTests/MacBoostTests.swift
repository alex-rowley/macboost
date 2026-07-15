import XCTest
@testable import MacBoost

// Behavioural test suite adapted from the LightGBM and XGBoost open-source
// test suites, restricted to features MacBoost implements (regression,
// binary logistic, dense float input). Upstream sources cited per test.
//
// GPU float atomics make histogram sums run-to-run nondeterministic (true of
// every GPU GBDT), so upstream exact-equality assertions on whole models are
// adapted into tolerance-based metric assertions where noted.
final class MacBoostTests: XCTestCase {

    // MARK: helpers

    func rmse(_ a: [Float], _ b: [Float]) -> Float {
        var s = 0.0
        for i in 0..<a.count { let d = Double(a[i] - b[i]); s += d * d }
        return Float((s / Double(a.count)).squareRoot())
    }

    func logloss(rawScores: [Float], labels: [Float]) -> Double {
        var s = 0.0
        for i in 0..<labels.count {
            let p = 1 / (1 + exp(-Double(rawScores[i])))
            let y = Double(labels[i])
            s += -(y * log(max(p, 1e-12)) + (1 - y) * log(max(1 - p, 1e-12)))
        }
        return s / Double(labels.count)
    }

    func fit(_ X: [Float], _ y: [Float], rows: Int, cols: Int,
             evalEvery: Int = 0, progress: ((String) -> Void)? = nil,
             configure: (inout BoosterParams) -> Void = { _ in }) throws -> MacBooster {
        var p = BoosterParams()
        configure(&p)
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y,
                  evalEvery: evalEvery, progress: progress)
        return b
    }

    /// Integer-valued features in 0..<levels, feature-major.
    func intData(rows: Int, cols: Int, levels: UInt64, seed: UInt64) -> [Float] {
        var X = [Float](repeating: 0, count: rows * cols)
        var rng = SplitMix64(seed: seed)
        for k in 0..<X.count { X[k] = Float(rng.next() % levels) }
        return X
    }

    // MARK: exact math

    /// Adapted from XGBoost's split-evaluator gtests: on a 4-point dataset
    /// the best split and the leaf values -lr * G / (H + lambda) are exact.
    func testExactSplitAndLeafValues() throws {
        let X: [Float] = [0, 0, 1, 1]
        let y: [Float] = [0, 0, 10, 10]
        let b = try fit(X, y, rows: 4, cols: 1) { p in
            p.numTrees = 1; p.maxDepth = 1; p.learningRate = 0.1
            p.lambda = 0; p.minChildHess = 1; p.minSplitGain = 0
        }
        // base = 5, gradients = [5,5,-5,-5]:
        // left leaf = -0.1 * 10/2 = -0.5, right leaf = +0.5
        XCTAssertEqual(b.baseScore, 5.0, accuracy: 1e-5)
        XCTAssertEqual(b.trees.count, 1)
        let tree = b.trees[0]
        XCTAssertEqual(tree.feature[0], 0, "root must split on the only feature")
        let pred = b.predict(featureMajor: [0, 1], rows: 2, cols: 1)
        XCTAssertEqual(pred[0], 4.5, accuracy: 1e-4)
        XCTAssertEqual(pred[1], 5.5, accuracy: 1e-4)
    }

    /// Adapted from XGBoost's evaluate_splits tests: the root split chosen on
    /// the GPU must match an exhaustive CPU search over every (feature,
    /// threshold), comparing achieved gain (robust to exact ties).
    func testRootSplitMatchesBruteForce() throws {
        let rows = 2000, cols = 5
        let X = intData(rows: rows, cols: cols, levels: 10, seed: 7)
        var rng = SplitMix64(seed: 8)
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows {
            y[i] = X[i] * X[i] - 3 * X[2 * rows + i] + 0.1 * rng.normal()
        }
        let lambda: Float = 1.0
        let b = try fit(X, y, rows: rows, cols: cols) { p in
            p.numTrees = 1; p.maxDepth = 1; p.lambda = lambda
        }
        let tree = b.trees[0]
        XCTAssertGreaterThanOrEqual(tree.feature[0], 0, "root must split")

        // Reference gain for split (f, x <= v), gradients g = base - y.
        var mean: Float = 0
        for v in y { mean += v }
        mean /= Float(rows)
        func gain(_ f: Int, _ v: Float) -> Double {
            var gl = 0.0, hl = 0.0, g = 0.0
            for i in 0..<rows {
                let gi = Double(mean - y[i])
                g += gi
                if X[f * rows + i] <= v { gl += gi; hl += 1 }
            }
            let hr = Double(rows) - hl, gr = g - gl
            guard hl >= 1, hr >= 1 else { return -.infinity }
            let lam = Double(lambda)
            return gl * gl / (hl + lam) + gr * gr / (hr + lam) - g * g / (Double(rows) + lam)
        }
        var bestGain = -Double.infinity
        for f in 0..<cols {
            for v in 0..<9 { bestGain = max(bestGain, gain(f, Float(v))) }
        }
        let chosenGain = gain(Int(tree.feature[0]), tree.threshold[0])
        XCTAssertEqual(chosenGain, bestGain, accuracy: max(1e-3 * bestGain, 1e-3),
                       "GPU-chosen root split must achieve the brute-force best gain")
    }

    // MARK: accuracy floors

    /// Adapted from LightGBM test_engine.py::test_regression (metric floor
    /// on a known dataset; Friedman #1 has noise floor 1.0).
    func testRegressionAccuracy() throws {
        let (X, y) = DataGen.friedman(rows: 50_000, cols: 10, seed: 1)
        let b = try fit(X, y, rows: 50_000, cols: 10) { p in
            p.numTrees = 100; p.maxDepth = 6
        }
        let (Xv, yv) = DataGen.friedman(rows: 10_000, cols: 10, seed: 2)
        let r = rmse(b.predict(featureMajor: Xv, rows: 10_000, cols: 10), yv)
        XCTAssertLessThan(r, 1.15, "valid RMSE should approach the 1.0 noise floor")
    }

    /// Adapted from LightGBM test_engine.py::test_binary (logloss floor).
    func testBinaryLogistic() throws {
        let rows = 20_000, cols = 4
        var rng = SplitMix64(seed: 3)
        var X = [Float](repeating: 0, count: rows * cols)
        for k in 0..<X.count { X[k] = rng.uniform() }
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows {
            y[i] = (X[i] + X[rows + i] + 0.3 * rng.normal()) > 1 ? 1 : 0
        }
        let b = try fit(X, y, rows: rows, cols: cols) { p in
            p.numTrees = 50; p.maxDepth = 5; p.objective = .binaryLogistic
        }
        let ll = logloss(rawScores: b.predict(featureMajor: X, rows: rows, cols: cols),
                         labels: y)
        // Bayes-optimal logloss for this noisy boundary is 0.407 (numeric
        // integration); LightGBM scores 0.385 on identical data with this
        // config (train, slight overfit). Anything <= 0.42 is at parity.
        XCTAssertLessThan(ll, 0.42, "logloss must reach the ~0.41 Bayes floor region")
    }

    /// Adapted from XGBoost test_basic training-continuation tests: more
    /// boosting rounds must not increase training loss.
    func testTrainLossDecreasesWithMoreTrees() throws {
        let (X, y) = DataGen.friedman(rows: 10_000, cols: 8, seed: 4)
        func trainRMSE(_ trees: Int) throws -> Float {
            let b = try fit(X, y, rows: 10_000, cols: 8) { p in
                p.numTrees = trees; p.maxDepth = 5
            }
            return rmse(b.predict(featureMajor: X, rows: 10_000, cols: 8), y)
        }
        let r5 = try trainRMSE(5), r50 = try trainRMSE(50)
        let baseline = rmse(y, [Float](repeating: y.reduce(0, +) / Float(y.count), count: y.count))
        XCTAssertLessThan(r5, baseline)
        XCTAssertLessThan(r50, r5)
    }

    // MARK: regularisation semantics

    /// Adapted from LightGBM/XGBoost parameter tests: larger L2 lambda must
    /// shrink leaf values.
    func testLambdaShrinksLeaves() throws {
        let (X, y) = DataGen.friedman(rows: 20_000, cols: 8, seed: 5)
        func maxAbsLeaf(_ lambda: Float) throws -> Float {
            let b = try fit(X, y, rows: 20_000, cols: 8) { p in
                p.numTrees = 10; p.maxDepth = 4; p.lambda = lambda
            }
            return b.trees.flatMap(\.leaf).map { abs($0) }.max() ?? 0
        }
        let loose = try maxAbsLeaf(0.001)
        let tight = try maxAbsLeaf(1000)
        XCTAssertLessThan(tight, loose)
    }

    /// Adapted from XGBoost min_child_weight tests: an unreachable hessian
    /// floor must prevent every split, leaving single-leaf trees.
    func testMinChildHessBlocksAllSplits() throws {
        let (X, y) = DataGen.friedman(rows: 5_000, cols: 6, seed: 6)
        let b = try fit(X, y, rows: 5_000, cols: 6) { p in
            p.numTrees = 5; p.maxDepth = 6; p.minChildHess = 5_001
        }
        for tree in b.trees {
            XCTAssertEqual(tree.feature[0], -1, "root must be a leaf")
        }
    }

    // MARK: degenerate inputs

    /// Adapted from LightGBM trivial-dataset tests: constant labels yield
    /// the constant prediction and no splits.
    func testConstantLabels() throws {
        let rows = 5_000, cols = 5
        var rng = SplitMix64(seed: 9)
        var X = [Float](repeating: 0, count: rows * cols)
        for k in 0..<X.count { X[k] = rng.uniform() }
        let y = [Float](repeating: 3.25, count: rows)
        let b = try fit(X, y, rows: rows, cols: cols) { p in
            p.numTrees = 10; p.maxDepth = 5
        }
        let pred = b.predict(featureMajor: X, rows: rows, cols: cols)
        for v in pred { XCTAssertEqual(v, 3.25, accuracy: 1e-4) }
        for tree in b.trees { XCTAssertEqual(tree.feature[0], -1) }
    }

    /// Rows far below the bin count must still train (LightGBM small-data
    /// handling tests).
    func testTinyDataset() throws {
        let rows = 50, cols = 3
        var rng = SplitMix64(seed: 10)
        var X = [Float](repeating: 0, count: rows * cols)
        for k in 0..<X.count { X[k] = rng.uniform() }
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows { y[i] = 2 * X[i] + rng.normal() * 0.1 }
        let b = try fit(X, y, rows: rows, cols: cols) { p in
            p.numTrees = 5; p.maxDepth = 3
        }
        let pred = b.predict(featureMajor: X, rows: rows, cols: cols)
        for v in pred { XCTAssertTrue(v.isFinite) }
        XCTAssertLessThan(rmse(pred, y), rmse(y, [Float](repeating: b.baseScore, count: rows)))
    }

    /// Adapted from LightGBM max_bin parameter tests: reduced bin counts
    /// still learn.
    func testReducedBinCount() throws {
        let (X, y) = DataGen.friedman(rows: 10_000, cols: 8, seed: 11)
        let b = try fit(X, y, rows: 10_000, cols: 8) { p in
            p.numTrees = 50; p.maxDepth = 5; p.numBins = 64
        }
        let (Xv, yv) = DataGen.friedman(rows: 5_000, cols: 8, seed: 12)
        XCTAssertLessThan(rmse(b.predict(featureMajor: Xv, rows: 5_000, cols: 8), yv), 1.3)
    }

    // MARK: invariances (tolerance-based: GPU float atomics are
    // run-to-run nondeterministic, so exact model equality is not required)

    /// Adapted from LightGBM feature-order invariance tests.
    func testFeaturePermutationInvariance() throws {
        let rows = 20_000, cols = 6
        let (X, y) = DataGen.friedman(rows: rows, cols: cols, seed: 13)
        var Xrev = [Float](repeating: 0, count: rows * cols)
        for f in 0..<cols {
            let src = f * rows, dst = (cols - 1 - f) * rows
            Xrev.replaceSubrange(dst..<(dst + rows), with: X[src..<(src + rows)])
        }
        let (Xv, yv) = DataGen.friedman(rows: 5_000, cols: cols, seed: 14)
        var XvRev = [Float](repeating: 0, count: 5_000 * cols)
        for f in 0..<cols {
            let src = f * 5_000, dst = (cols - 1 - f) * 5_000
            XvRev.replaceSubrange(dst..<(dst + 5_000), with: Xv[src..<(src + 5_000)])
        }
        func run(_ Xt: [Float], _ Xvt: [Float]) throws -> Float {
            let b = try fit(Xt, y, rows: rows, cols: cols) { p in
                p.numTrees = 50; p.maxDepth = 5
            }
            return rmse(b.predict(featureMajor: Xvt, rows: 5_000, cols: cols), yv)
        }
        let r1 = try run(X, Xv), r2 = try run(Xrev, XvRev)
        XCTAssertEqual(r1, r2, accuracy: 0.03)
    }

    /// Refit reproducibility (upstream: deterministic-training tests,
    /// adapted to metric tolerance for GPU atomics).
    func testRefitReproducibility() throws {
        let rows = 20_000, cols = 6
        let (X, y) = DataGen.friedman(rows: rows, cols: cols, seed: 15)
        let (Xv, yv) = DataGen.friedman(rows: 5_000, cols: cols, seed: 16)
        func run() throws -> Float {
            let b = try fit(X, y, rows: rows, cols: cols) { p in
                p.numTrees = 50; p.maxDepth = 5
            }
            return rmse(b.predict(featureMajor: Xv, rows: 5_000, cols: cols), yv)
        }
        let r1 = try run(), r2 = try run()
        XCTAssertEqual(r1, r2, accuracy: 0.02)
    }

    // MARK: internal consistency

    /// The train metric computed from the GPU's running predictions (binned
    /// routing) must match CPU predict on the training data (raw-threshold
    /// routing) — verifies bin/threshold equivalence. Upstream analogue:
    /// XGBoost test_basic.py::test_predict on the training matrix.
    func testTrainMetricMatchesPredict() throws {
        let rows = 20_000, cols = 8
        let (X, y) = DataGen.friedman(rows: rows, cols: cols, seed: 17)
        var reported: Float = -1
        let b = try fit(X, y, rows: rows, cols: cols, evalEvery: 50, progress: { line in
            if let last = line.split(separator: " ").last, let v = Float(last) {
                reported = v
            }
        }) { p in
            p.numTrees = 50; p.maxDepth = 5
        }
        XCTAssertGreaterThan(reported, 0, "progress callback must report train RMSE")
        let recomputed = rmse(b.predict(featureMajor: X, rows: rows, cols: cols), y)
        XCTAssertEqual(reported, recomputed, accuracy: 2e-3)
    }
}
