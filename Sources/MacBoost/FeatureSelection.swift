import Foundation
import Metal

// Boruta feature selection (Kursa & Rudnicki 2010), GPU-resident: each
// round trains on the real features plus "shadow" copies whose rows are
// permuted, and a feature scores a hit when its gain importance beats the
// best shadow. Shadows never exist as data — a kernel gathers each
// column's bin bytes through a per-column random bijection directly into
// a double-width binned matrix in GPU memory (bin edges are permutation-
// invariant), and a fresh seed re-permutes them each round.

private struct ShadowParamsHost {
    var numSamples: UInt32; var numFeatures: UInt32
    var halfBits: UInt32; var seed: UInt32
}

public struct FeatureSelectionResult {
    /// Features whose gain beat the best shadow significantly often.
    public let confirmed: [Int]
    /// Not significant either way at the given alpha (Boruta's "tentative").
    public let tentative: [Int]
    /// Features that significantly failed to beat the best shadow.
    public let rejected: [Int]
    /// Rounds (out of `rounds`) each feature beat the best shadow.
    public let hits: [Int]
    public let rounds: Int
    /// Mean gain importance relative to the mean best-shadow gain; > 1
    /// means the feature out-gained the noise ceiling on average.
    public let gainRatio: [Float]
}

extension MacBooster {

    /// Boruta shadow-feature selection. Trains `rounds` disposable models
    /// on [X | shadow(X)] entirely in GPU memory and classifies every
    /// feature as confirmed / tentative / rejected with a Bonferroni-
    /// corrected binomial test. Use the result via
    /// `params.allowedFeatures` on a fresh booster (the final model then
    /// never splits on rejected features but still accepts full-width X).
    ///
    /// `trees` sizes the disposable probe models. Boruta needs gain-vs-
    /// shadow votes, not converged ensembles, so the probes default to
    /// min(numTrees, 100) boosting rounds rather than the full spec —
    /// pass an explicit count to override either way.
    public func selectFeatures(featureMajor X: [Float], rows: Int, cols: Int,
                               labels: [Float], weights: [Float]? = nil,
                               rounds: Int = 20, trees: Int? = nil,
                               alpha: Double = 0.05,
                               seed: UInt64 = 0,
                               progress: ((String) -> Void)? = nil) throws
        -> FeatureSelectionResult {
        guard rows > 0 && cols > 0, X.count == rows * cols else {
            throw MacBoostError.invalidInput(
                "selection data shapes inconsistent (rows=\(rows), cols=\(cols), X=\(X.count))")
        }
        guard labels.count == rows else {
            throw MacBoostError.invalidInput("labels has \(labels.count) rows, X has \(rows)")
        }
        guard rounds >= 5 else {
            throw MacBoostError.invalidInput("selection needs rounds >= 5 for the binomial test")
        }
        guard alpha > 0 && alpha < 1 else {
            throw MacBoostError.invalidInput("alpha must be in (0, 1)")
        }
        if let trees {
            guard trees >= 1 else {
                throw MacBoostError.invalidInput("selection trees must be >= 1")
            }
        }
        try validateLabels(labels)
        try X.withUnsafeBufferPointer { xp in
            try validateCategoricals(
                MatrixView(base: xp.baseAddress!, rows: rows, cols: cols,
                           rowMajor: false), rows: rows, cols: cols)
        }

        // Bin the real features once; every round reuses these bytes.
        let nBins = params.numBins
        let categorical = params.categoricalFeatures
        var featFlags = [UInt8](repeating: 0, count: cols)
        for f in categorical { featFlags[f] = 1 }
        let edges = Binner.computeEdges(featureMajor: X, rows: rows, cols: cols,
                                        numBins: nBins, categorical: categorical)
        let numTiles = (cols + 7) / 8
        let binsBuf = engine.makeBuffer(length: rows * numTiles * 8)
        do {
            let cb = engine.queue.makeCommandBuffer()!
            engine.dispatch(cb, "bin_data",
                            buffers: [engine.makeBuffer(X), engine.makeBuffer(edges),
                                      binsBuf, engine.makeBuffer(featFlags)],
                            params: BinParams(numSamples: UInt32(rows),
                                              numFeatures: UInt32(cols),
                                              numBins: UInt32(nBins),
                                              rowStride: 1,
                                              colStride: UInt32(rows)),
                            grid: MTLSize(width: rows, height: cols, depth: 1),
                            threadgroup: MTLSize(width: 256, height: 1, depth: 1))
            cb.commit()
            cb.waitUntilCompleted()
        }

        // Double-width shadow matrix, refilled in place each round.
        let cols2 = 2 * cols
        let edges2 = edges + edges
        var categorical2 = categorical
        for f in categorical { categorical2.insert(f + cols) }
        let numTiles2 = (cols2 + 7) / 8
        let bins2Buf = engine.makeBuffer(length: rows * numTiles2 * 8)
        var bits = 1
        while (1 << bits) < rows { bits += 1 }
        let halfBits = UInt32((bits + 1) / 2)

        var selParams = params
        selParams.monotoneConstraints = nil
        selParams.allowedFeatures = nil
        selParams.categoricalFeatures = categorical2
        selParams.numTrees = trees ?? min(params.numTrees, 100)

        var hits = [Int](repeating: 0, count: cols)
        var gainSum = [Double](repeating: 0, count: cols)
        var shadowSum = 0.0
        for round in 0..<rounds {
            // Each round creates thousands of autoreleased Metal objects
            // (one command buffer per tree); drain them per round or a
            // long selection run on a big multiclass spec gets jetsammed.
            let gains = try autoreleasepool { () -> [Float] in
                let roundSeed = UInt32(truncatingIfNeeded:
                    seed &+ UInt64(round) &* 0x9E3779B97F4A7C15 &+ 1)
                let cb = engine.queue.makeCommandBuffer()!
                engine.dispatch(cb, "shadow_bins",
                                buffers: [binsBuf, bins2Buf],
                                params: ShadowParamsHost(numSamples: UInt32(rows),
                                                         numFeatures: UInt32(cols),
                                                         halfBits: halfBits,
                                                         seed: roundSeed),
                                grid: MTLSize(width: rows, height: cols, depth: 1),
                                threadgroup: MTLSize(width: 256, height: 1, depth: 1))
                cb.commit()
                cb.waitUntilCompleted()

                let sb = try MacBooster(params: selParams, engine: engine)
                try sb.fitImpl(X: nil, prebinned: nil,
                               gpuBinned: GPUBinned(bins: bins2Buf, edges: edges2,
                                                    categorical: categorical2,
                                                    numBins: nBins),
                               rows: rows, cols: cols2, labels: labels,
                               weights: weights, valid: nil,
                               earlyStoppingRounds: 0, evalEvery: 0,
                               initModel: nil, progress: nil)
                return sb.featureImportance(type: .gain)
            }
            let maxShadow = gains[cols...].max() ?? 0
            for f in 0..<cols where gains[f] > maxShadow { hits[f] += 1 }
            for f in 0..<cols { gainSum[f] += Double(gains[f]) }
            shadowSum += Double(maxShadow)
            if let progress {
                let leading = hits.filter { $0 == round + 1 }.count
                progress("[Selection] round \(round + 1)/\(rounds): "
                         + "\(leading) features beat every shadow so far")
            }
        }

        // Boruta's verdicts: two one-sided binomial tests against p = 0.5,
        // Bonferroni-corrected across features.
        let cutoff = alpha / Double(cols)
        var confirmed: [Int] = [], tentative: [Int] = [], rejected: [Int] = []
        for f in 0..<cols {
            if binomialTail(atLeast: hits[f], rounds: rounds) < cutoff {
                confirmed.append(f)
            } else if binomialTail(atLeast: rounds - hits[f], rounds: rounds) < cutoff {
                rejected.append(f)
            } else {
                tentative.append(f)
            }
        }
        let ratio = (0..<cols).map {
            Float(gainSum[$0] / max(shadowSum, .leastNormalMagnitude))
        }
        progress?("[Selection] confirmed \(confirmed.count), "
                  + "tentative \(tentative.count), rejected \(rejected.count) "
                  + "of \(cols) features")
        return FeatureSelectionResult(confirmed: confirmed, tentative: tentative,
                                      rejected: rejected, hits: hits,
                                      rounds: rounds, gainRatio: ratio)
    }
}

/// P[Binomial(rounds, 0.5) >= k], exact.
private func binomialTail(atLeast k: Int, rounds: Int) -> Double {
    if k <= 0 { return 1 }
    // pmf(0) = 0.5^rounds, pmf(i+1) = pmf(i) * (rounds-i)/(i+1)
    var pmf = pow(0.5, Double(rounds))
    var tail = 0.0
    for i in 0...rounds {
        if i >= k { tail += pmf }
        if i < rounds { pmf *= Double(rounds - i) / Double(i + 1) }
    }
    return min(1, tail)
}
