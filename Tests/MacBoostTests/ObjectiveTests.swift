import XCTest
@testable import MacBoost

/// Objectives pack + sample weights + scale_pos_weight, with semantic
/// assertions (quantile coverage, weight/duplication equivalence) rather
/// than just "loss went down".
final class ObjectiveTests: XCTestCase {

    func rmse(_ a: [Float], _ b: [Float]) -> Float {
        var s = 0.0
        for i in 0..<a.count { let d = Double(a[i] - b[i]); s += d * d }
        return Float((s / Double(a.count)).squareRoot())
    }

    func fitPredict(_ obj: Objective, _ X: [Float], _ y: [Float], rows: Int, cols: Int,
                    trees: Int = 150, configure: (inout BoosterParams) -> Void = { _ in })
        throws -> (MacBooster, [Float]) {
        var p = BoosterParams()
        p.numTrees = trees; p.maxDepth = 5; p.objective = obj
        configure(&p)
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y)
        return (b, b.predict(featureMajor: X, rows: rows, cols: cols))
    }

    /// MAE and Huber must be robust to label outliers where L2 is not.
    func testMAEAndHuberRobustToOutliers() throws {
        let rows = 20_000
        var (X, y) = DataGen.friedman(rows: rows, cols: 8, seed: 81)
        var rng = SplitMix64(seed: 82)
        var yClean = y
        for i in 0..<rows where rng.uniform() < 0.02 { y[i] += 500 }   // gross outliers
        _ = yClean

        func medianError(_ obj: Objective) throws -> Float {
            let (_, pred) = try fitPredict(obj, X, y, rows: rows, cols: 8)
            var absErr = (0..<rows).map { abs(pred[$0] - yClean[$0]) }
            absErr.sort()
            return absErr[rows / 2]
        }
        let l2 = try medianError(.regression)
        let mae = try medianError(.mae)
        let huber = try medianError(.huber)
        XCTAssertLessThan(mae, l2, "L1 must resist outliers better than L2")
        XCTAssertLessThan(huber, l2, "Huber must resist outliers better than L2")
    }

    /// Quantile regression's defining property: empirical coverage matches
    /// alpha (adapted from LightGBM's quantile objective tests).
    func testQuantileCoverage() throws {
        let rows = 30_000
        let (X, y) = DataGen.friedman(rows: rows, cols: 8, seed: 83)
        for alpha in [Float(0.1), 0.9] {
            let (_, pred) = try fitPredict(.quantile, X, y, rows: rows, cols: 8) { p in
                p.alpha = alpha
            }
            // Leaf renewal (residual quantiles per leaf) keeps extreme
            // quantiles calibrated, matching LightGBM's behaviour.
            let coverage = Float((0..<rows).filter { y[$0] <= pred[$0] }.count) / Float(rows)
            XCTAssertEqual(coverage, alpha, accuracy: 0.03,
                           "alpha=\(alpha) coverage was \(coverage)")
        }
    }

    /// Poisson with log link: predictions are positive means approximating
    /// the true rate.
    func testPoisson() throws {
        let rows = 30_000, cols = 5
        var rng = SplitMix64(seed: 84)
        var X = [Float](repeating: 0, count: rows * cols)
        for k in 0..<X.count { X[k] = rng.uniform() }
        var y = [Float](repeating: 0, count: rows)
        var rates = [Float](repeating: 0, count: rows)
        for i in 0..<rows {
            let rate = exp(0.5 + 1.5 * X[i])            // in [1.6, 7.4]
            rates[i] = rate
            // Knuth poisson sampling
            var l = exp(-Double(rate)), p = 1.0, k = -1
            repeat { k += 1; p *= Double(rng.uniform()) } while p > l
            y[i] = Float(k)
        }
        let (_, pred) = try fitPredict(.poisson, X, y, rows: rows, cols: cols)
        XCTAssertTrue(pred.allSatisfy { $0 > 0 }, "poisson predictions are means, must be positive")
        XCTAssertLessThan(rmse(pred, rates), 0.5,
                          "predicted means must approximate the true rates")
    }

    /// Tweedie on zero-inflated positive data (the insurance shape).
    func testTweedie() throws {
        let rows = 30_000, cols = 5
        var rng = SplitMix64(seed: 85)
        var X = [Float](repeating: 0, count: rows * cols)
        for k in 0..<X.count { X[k] = rng.uniform() }
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows {
            if rng.uniform() < 0.6 { y[i] = 0 }          // zero-inflated
            else { y[i] = exp(1.0 + 2.0 * X[i]) * (0.5 + rng.uniform()) }
        }
        let (_, pred) = try fitPredict(.tweedie, X, y, rows: rows, cols: cols) { p in
            p.tweedieVariancePower = 1.5
        }
        XCTAssertTrue(pred.allSatisfy { $0 > 0 })
        // Predictions approximate E[y|x] = 0.4 * exp(1 + 2 x0)
        var s = 0.0
        for i in 0..<rows { s += Double(pred[i] - 0.4 * exp(1.0 + 2.0 * X[i])) }
        XCTAssertEqual(Float(s / Double(rows)), 0, accuracy: 0.4,
                       "tweedie means must be nearly unbiased")
    }

    /// The classic weights identity: weight w=2 on a subset must behave
    /// like duplicating those rows (adapted from LightGBM's weight tests).
    func testWeightsEquivalentToDuplication() throws {
        let rows = 10_000, cols = 6
        let (X, y) = DataGen.friedman(rows: rows, cols: cols, seed: 86)
        let (Xv, yv) = DataGen.friedman(rows: 5_000, cols: cols, seed: 87)

        // Weighted fit: first half weight 2, second half weight 1.
        var weights = [Float](repeating: 1, count: rows)
        for i in 0..<(rows / 2) { weights[i] = 2 }
        var p = BoosterParams(); p.numTrees = 80; p.maxDepth = 5
        let bw = try MacBooster(params: p)
        try bw.fit(featureMajor: X, rows: rows, cols: cols, labels: y, weights: weights)

        // Duplicated fit: first half appears twice (feature-major append).
        let dupRows = rows + rows / 2
        var Xd = [Float](); Xd.reserveCapacity(dupRows * cols)
        for f in 0..<cols {
            Xd.append(contentsOf: X[(f * rows)..<(f * rows + rows)])
            Xd.append(contentsOf: X[(f * rows)..<(f * rows + rows / 2)])
        }
        // Feature-major layout: regroup so each feature's column is contiguous.
        var Xd2 = [Float](repeating: 0, count: dupRows * cols)
        for f in 0..<cols {
            for i in 0..<rows { Xd2[f * dupRows + i] = X[f * rows + i] }
            for i in 0..<(rows / 2) { Xd2[f * dupRows + rows + i] = X[f * rows + i] }
        }
        var yd = y; yd.append(contentsOf: y[0..<(rows / 2)])
        let bd = try MacBooster(params: p)
        try bd.fit(featureMajor: Xd2, rows: dupRows, cols: cols, labels: yd)
        _ = Xd

        let rw = rmse(bw.predict(featureMajor: Xv, rows: 5_000, cols: cols), yv)
        let rd = rmse(bd.predict(featureMajor: Xv, rows: 5_000, cols: cols), yv)
        XCTAssertEqual(rw, rd, accuracy: 0.04,
                       "weight=2 must be equivalent to row duplication (got \(rw) vs \(rd))")
    }

    /// scale_pos_weight must shift predicted probabilities upward on
    /// imbalanced data.
    func testScalePosWeight() throws {
        let rows = 20_000, cols = 4
        var rng = SplitMix64(seed: 88)
        var X = [Float](repeating: 0, count: rows * cols)
        for k in 0..<X.count { X[k] = rng.uniform() }
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows where X[i] > 0.9 && rng.uniform() < 0.5 { y[i] = 1 }  // ~5% positive

        // In the positive region (x0 > 0.9, true rate ~0.5) the weighted
        // optimum is spw*p/(spw*p + 1-p): ~0.5 at spw=1, ~0.91 at spw=10.
        func regionProba(_ spw: Float) throws -> Float {
            var p = BoosterParams()
            p.numTrees = 60; p.maxDepth = 4; p.objective = .binaryLogistic
            p.scalePosWeight = spw
            let b = try MacBooster(params: p)
            try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y)
            let raw = b.predict(featureMajor: X, rows: rows, cols: cols)
            var s = 0.0, n = 0.0
            for i in 0..<rows where X[i] > 0.92 {
                s += 1 / (1 + exp(-Double(raw[i]))); n += 1
            }
            return Float(s / n)
        }
        let base = try regionProba(1)
        let boosted = try regionProba(10)
        XCTAssertGreaterThan(boosted, base + 0.2,
                             "spw=10 must lift positive-region probability toward "
                             + "spw*p/(spw*p+1-p) (got \(base) -> \(boosted))")
    }

    /// Invalid inputs for the new objectives are rejected.
    func testObjectiveValidation() throws {
        var p = BoosterParams(); p.objective = .quantile; p.alpha = 1.5
        XCTAssertThrowsError(try MacBooster(params: p))
        var p2 = BoosterParams(); p2.objective = .tweedie; p2.tweedieVariancePower = 2.5
        XCTAssertThrowsError(try MacBooster(params: p2))
        var p3 = BoosterParams(); p3.objective = .poisson; p3.numTrees = 5
        let b = try MacBooster(params: p3)
        XCTAssertThrowsError(try b.fit(featureMajor: [1, 2, 3, 4], rows: 4, cols: 1,
                                       labels: [1, -2, 3, 4])) { err in
            guard case MacBoostError.invalidInput = err else {
                return XCTFail("expected invalidInput, got \(err)")
            }
        }
    }
}

/// Batch 2: sampling knobs, importance, cover.
extension ObjectiveTests {

    /// Bagging must actually sample (tiny subsample degrades) while
    /// reasonable subsample stays near full accuracy.
    func testBaggingSubsample() throws {
        let (X, y) = DataGen.friedman(rows: 30_000, cols: 8, seed: 91)
        let (Xv, yv) = DataGen.friedman(rows: 5_000, cols: 8, seed: 92)
        func run(_ sub: Float) throws -> Float {
            var p = BoosterParams(); p.numTrees = 100; p.maxDepth = 5; p.subsample = sub
            let b = try MacBooster(params: p)
            try b.fit(featureMajor: X, rows: 30_000, cols: 8, labels: y)
            return rmse(b.predict(featureMajor: Xv, rows: 5_000, cols: 8), yv)
        }
        let full = try run(1.0)
        let bagged = try run(0.7)
        let tiny = try run(0.02)
        XCTAssertLessThan(bagged, full + 0.08, "subsample=0.7 stays near full accuracy")
        XCTAssertGreaterThan(tiny, full + 0.05,
                             "subsample=0.02 must visibly degrade — otherwise bagging is a no-op")
        var bad = BoosterParams(); bad.goss = true; bad.subsample = 0.5
        XCTAssertThrowsError(try MacBooster(params: bad), "goss + subsample must be rejected")
    }

    /// colsample_bytree: masked features may not be split on.
    func testFeatureFraction() throws {
        let (X, y) = DataGen.friedman(rows: 20_000, cols: 10, seed: 93)
        var p = BoosterParams(); p.numTrees = 80; p.maxDepth = 5; p.featureFraction = 0.5
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: 20_000, cols: 10, labels: y)
        // With half the features per tree the model must still learn well
        // (friedman has 5 informative of 10).
        let (Xv, yv) = DataGen.friedman(rows: 5_000, cols: 10, seed: 94)
        XCTAssertLessThan(rmse(b.predict(featureMajor: Xv, rows: 5_000, cols: 10), yv), 1.4)
        // Different trees must use different feature subsets: collect split
        // features per tree and check at least two trees differ.
        var sets = Set<Set<Int32>>()
        for tree in b.trees.prefix(20) {
            var used = Set<Int32>()
            for n in 0..<tree.feature.count where tree.feature[n] >= 0 {
                used.insert(tree.feature[n])
            }
            sets.insert(used)
        }
        XCTAssertGreaterThan(sets.count, 1, "per-tree feature subsets must vary")
    }

    /// Gain importance must concentrate on the 5 informative friedman
    /// features (adapted from LightGBM's feature_importance tests).
    func testFeatureImportance() throws {
        let (X, y) = DataGen.friedman(rows: 30_000, cols: 10, seed: 95)
        var p = BoosterParams(); p.numTrees = 100; p.maxDepth = 5
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: 30_000, cols: 10, labels: y)
        for type in [MacBooster.ImportanceType.gain, .split] {
            let imp = b.featureImportance(type: type)
            let informative = imp[0..<5].reduce(0, +)
            let noise = imp[5..<10].reduce(0, +)
            XCTAssertGreaterThan(informative, noise * 5,
                                 "\(type) importance must concentrate on informative features")
        }
    }

    /// Cover: the root covers every sample; children partition the parent.
    func testCoverConsistency() throws {
        let rows = 10_000
        let (X, y) = DataGen.friedman(rows: rows, cols: 6, seed: 96)
        var p = BoosterParams(); p.numTrees = 5; p.maxDepth = 5
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: 6, labels: y)
        for tree in b.trees {
            XCTAssertEqual(tree.cover[0], Float(rows), "root cover = all samples")
            for n in 0..<((tree.cover.count - 1) / 2) where tree.feature[n] >= 0 {
                XCTAssertEqual(tree.cover[2 * n + 1] + tree.cover[2 * n + 2], tree.cover[n],
                               accuracy: 0.5, "children partition the parent's samples")
            }
        }
    }
}

/// Batch 3: monotone constraints, AUC metric, warm starts.
extension ObjectiveTests {

    /// A +1 constraint on x3 (true coefficient +10) must make predictions
    /// non-decreasing along x3 for every probe row (adapted from LightGBM's
    /// test_monotone_constraints).
    func testMonotoneConstraint() throws {
        let rows = 30_000
        let (X, y) = DataGen.friedman(rows: rows, cols: 6, seed: 101)
        var p = BoosterParams()
        p.numTrees = 150; p.maxDepth = 6
        p.monotoneConstraints = [0, 0, 0, 1, 0, 0]
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: 6, labels: y)

        var rng = SplitMix64(seed: 102)
        let steps = 30
        var violations = 0
        for _ in 0..<50 {                       // 50 random rows swept along x3
            var probe = (0..<6).map { _ in rng.uniform() }
            var prev = -Float.infinity
            for s in 0..<steps {
                probe[3] = Float(s) / Float(steps - 1)
                var Xp = [Float](repeating: 0, count: 6)
                for f in 0..<6 { Xp[f] = probe[f] }
                let v = b.predict(featureMajor: Xp, rows: 1, cols: 6)[0]
                if v < prev - 1e-5 { violations += 1 }
                prev = v
            }
        }
        let allLeaves = b.trees.flatMap(\.leaf)
        print("DEBUG final: finite \(allLeaves.filter { $0 == -666 }.count), " +
              "inf \(allLeaves.filter { $0 == -555 }.count)")
        // Structural audit: at every x3 split, all left-subtree leaves must
        // be <= all right-subtree leaves (the hard guarantee bounds provide).
        for tr in b.trees {
            func leafRange(_ root: Int) -> (Float, Float) {
                var mn = Float.infinity, mx = -Float.infinity
                var stack = [root]
                while let n = stack.popLast() {
                    if tr.feature[n] >= 0 { stack += [2 * n + 1, 2 * n + 2] }
                    else if tr.cover[n] > 0 { mn = min(mn, tr.leaf[n]); mx = max(mx, tr.leaf[n]) }
                }
                return (mn, mx)
            }
            for n in 0..<tr.feature.count where tr.feature[n] == 3 {
                let (_, lmax) = leafRange(2 * n + 1)
                let (rmin, _) = leafRange(2 * n + 2)
                XCTAssertLessThanOrEqual(lmax, rmin + 1e-6,
                                         "left subtree must not exceed right subtree")
            }
        }
        XCTAssertEqual(violations, 0, "monotone(+1) on x3 must never decrease along x3")
        // Accuracy must not be destroyed by the (satisfiable) constraint.
        let (Xv, yv) = DataGen.friedman(rows: 5_000, cols: 6, seed: 103)
        XCTAssertLessThan(rmse(b.predict(featureMajor: Xv, rows: 5_000, cols: 6), yv), 1.3)
    }

    /// AUC as the validation metric: reported value matches a reference
    /// computation and early stopping maximises rather than minimises.
    func testAUCMetricAndEarlyStopping() throws {
        let rows = 10_000, cols = 4
        var rng = SplitMix64(seed: 104)
        var X = [Float](repeating: 0, count: rows * cols)
        for k in 0..<X.count { X[k] = rng.uniform() }
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows {
            y[i] = (X[i] + X[rows + i] + 0.5 * rng.normal()) > 1 ? 1 : 0
        }
        var p = BoosterParams()
        p.numTrees = 200; p.maxDepth = 5; p.objective = .binaryLogistic
        p.metric = .auc; p.learningRate = 0.3
        let b = try MacBooster(params: p)
        let (Xv, yv) = (Array(X.prefix(2_000 * cols)), [Float](y.prefix(2_000)))
        // A proper valid set: regenerate to avoid feature-major slicing issues.
        var Xvv = [Float](repeating: 0, count: 2_000 * cols)
        var yvv = [Float](repeating: 0, count: 2_000)
        for i in 0..<2_000 {
            for f in 0..<cols { Xvv[f * 2_000 + i] = rng.uniform() }
        }
        for i in 0..<2_000 {
            yvv[i] = (Xvv[i] + Xvv[2_000 + i] + 0.5 * rng.normal()) > 1 ? 1 : 0
        }
        _ = (Xv, yv)
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y,
                  valid: EvalSet(featureMajor: Xvv, rows: 2_000, labels: yvv),
                  earlyStoppingRounds: 25)
        guard let best = b.bestIteration else { return XCTFail("bestIteration unset") }
        XCTAssertEqual(b.trees.count, best)
        // Reported AUC must be sane and high for a separable-ish problem.
        let bestAUC = b.evalHistory[best - 1]
        XCTAssertGreaterThan(bestAUC, 0.75)
        XCTAssertEqual(bestAUC, b.evalHistory.max()!,
                       "early stopping must have kept the MAXIMUM AUC iteration")
    }

    /// Warm start: continuing from an init model must behave like training
    /// the full tree count in one go.
    func testContinuedTraining() throws {
        let rows = 20_000
        let (X, y) = DataGen.friedman(rows: rows, cols: 8, seed: 105)
        let (Xv, yv) = DataGen.friedman(rows: 5_000, cols: 8, seed: 106)
        var p = BoosterParams(); p.numTrees = 50; p.maxDepth = 5
        let first = try MacBooster(params: p)
        try first.fit(featureMajor: X, rows: rows, cols: 8, labels: y)
        let second = try MacBooster(params: p)
        try second.fit(featureMajor: X, rows: rows, cols: 8, labels: y, initModel: first)
        XCTAssertEqual(second.trees.count, 100, "50 init + 50 new trees")

        var pFull = BoosterParams(); pFull.numTrees = 100; pFull.maxDepth = 5
        let full = try MacBooster(params: pFull)
        try full.fit(featureMajor: X, rows: rows, cols: 8, labels: y)
        let rCont = rmse(second.predict(featureMajor: Xv, rows: 5_000, cols: 8), yv)
        let rFull = rmse(full.predict(featureMajor: Xv, rows: 5_000, cols: 8), yv)
        XCTAssertEqual(rCont, rFull, accuracy: 0.03,
                       "warm start must match one-shot training (\(rCont) vs \(rFull))")
    }
}

/// Batch 4: TreeSHAP contributions.
extension ObjectiveTests {

    /// The defining TreeSHAP property (LightGBM tests this for
    /// pred_contrib): per-row contributions + expected value == the raw
    /// prediction. Covers numeric + missing + categorical routing.
    func testSHAPSumsToPrediction() throws {
        let rows = 1_500, cols = 7
        var (X, y) = DataGen.friedman(rows: rows, cols: 6, seed: 111)
        var rng = SplitMix64(seed: 112)
        for i in 0..<rows where rng.uniform() < 0.1 { X[rows + i] = .nan }
        X.append(contentsOf: (0..<rows).map { _ in Float(rng.next() % 5) })
        for i in 0..<rows { y[i] += X[6 * rows + i] == 2 ? 3 : 0 }

        var p = BoosterParams()
        p.numTrees = 40; p.maxDepth = 6; p.categoricalFeatures = [6]
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y)

        let contrib = b.predictContributions(featureMajor: X, rows: rows, cols: cols)
        let raw = b.predict(featureMajor: X, rows: rows, cols: cols)
        var worst: Float = 0
        for i in 0..<rows {
            var s: Float = 0
            for k in 0...cols { s += contrib[i * (cols + 1) + k] }
            worst = max(worst, abs(s - raw[i]))
        }
        XCTAssertLessThan(worst, 2e-2,
                          "SHAP contributions must sum to the prediction (worst gap \(worst))")

        // Attribution must concentrate on informative features (0-4, 6).
        var mass = [Float](repeating: 0, count: cols)
        for i in 0..<rows {
            for f in 0..<cols { mass[f] += abs(contrib[i * (cols + 1) + f]) }
        }
        XCTAssertGreaterThan(mass[3], mass[5] * 5,
                             "x3 (coef 10) must out-attribute the noise feature")
        XCTAssertGreaterThan(mass[6], mass[5] * 2,
                             "the categorical signal must out-attribute noise")
    }

    /// Multiclass SHAP: per-class blocks sum to the class's raw score.
    func testMulticlassSHAP() throws {
        let rows = 800, cols = 4, K = 3
        var rng = SplitMix64(seed: 131)
        var X = [Float](repeating: 0, count: rows * cols)
        for k in 0..<X.count { X[k] = rng.uniform() }
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows { y[i] = Float(min(2, Int(X[i] * 3))) }
        var p = BoosterParams()
        p.numTrees = 25; p.maxDepth = 4
        p.objective = .multiclass; p.numClasses = K
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y)

        let contrib = b.predictContributions(featureMajor: X, rows: rows, cols: cols)
        let raw = b.predict(featureMajor: X, rows: rows, cols: cols)
        var worst: Float = 0
        for i in 0..<rows {
            for k in 0..<K {
                var s: Float = 0
                for f in 0...cols { s += contrib[i * K * (cols + 1) + k * (cols + 1) + f] }
                worst = max(worst, abs(s - raw[i * K + k]))
            }
        }
        XCTAssertLessThan(worst, 2e-2,
                          "per-class contributions must sum to the class score (worst \(worst))")
    }
}

/// Batch 5: multiclass softmax.
extension ObjectiveTests {

    func testMulticlass() throws {
        let rows = 20_000, cols = 5, K = 4
        var rng = SplitMix64(seed: 121)
        var X = [Float](repeating: 0, count: rows * cols)
        for k in 0..<X.count { X[k] = rng.uniform() }
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows {
            // 4 quadrant-ish classes from x0, x1 with noise.
            let a = X[i] + 0.15 * rng.normal() > 0.5 ? 1 : 0
            let b = X[rows + i] + 0.15 * rng.normal() > 0.5 ? 2 : 0
            y[i] = Float(a + b)
        }
        var p = BoosterParams()
        p.numTrees = 60; p.maxDepth = 5
        p.objective = .multiclass; p.numClasses = K
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: cols, labels: y)
        XCTAssertEqual(b.trees.count, 60 * K, "K trees per boosting round")

        let raw = b.predict(featureMajor: X, rows: rows, cols: cols)
        XCTAssertEqual(raw.count, rows * K, "multiclass predict returns rows x K")
        var correct = 0
        for i in 0..<rows {
            var best = 0
            for c in 1..<K where raw[i * K + c] > raw[i * K + best] { best = c }
            if Float(best) == y[i] { correct += 1 }
        }
        let acc = Float(correct) / Float(rows)
        XCTAssertGreaterThan(acc, 0.74, "accuracy at the ~0.775 Bayes ceiling (acc \(acc))")

        // GPU and CPU multiclass predictions must agree exactly.
        let cpu = b.predictCPU(featureMajor: X, rows: rows, cols: cols)
        XCTAssertEqual(raw, cpu, "multiclass GPU forest must match CPU walk")

        // Round-trip.
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("macboost-mc-\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: url) }
        try b.save(to: url)
        let loaded = try MacBooster.load(from: url)
        XCTAssertEqual(loaded.predict(featureMajor: X, rows: rows, cols: cols), raw)

        // Early stopping in rounds with a valid set.
        var Xv = [Float](repeating: 0, count: 4_000 * cols)
        var yv = [Float](repeating: 0, count: 4_000)
        for i in 0..<4_000 {
            for f in 0..<cols { Xv[f * 4_000 + i] = rng.uniform() }
            let a = Xv[i] + 0.15 * rng.normal() > 0.5 ? 1 : 0
            let bb = Xv[4_000 + i] + 0.15 * rng.normal() > 0.5 ? 2 : 0
            yv[i] = Float(a + bb)
        }
        var p2 = p; p2.numTrees = 300; p2.learningRate = 0.4
        let es = try MacBooster(params: p2)
        try es.fit(featureMajor: X, rows: rows, cols: cols, labels: y,
                   valid: EvalSet(featureMajor: Xv, rows: 4_000, labels: yv),
                   earlyStoppingRounds: 15)
        guard let best = es.bestIteration else { return XCTFail() }
        XCTAssertEqual(es.trees.count, best * K,
                       "truncation must respect round boundaries")

        // Bad labels rejected.
        var p3 = p; p3.numTrees = 2
        let bad = try MacBooster(params: p3)
        XCTAssertThrowsError(try bad.fit(featureMajor: X, rows: rows, cols: cols,
                                         labels: y.map { $0 + 0.5 }))
    }
}
