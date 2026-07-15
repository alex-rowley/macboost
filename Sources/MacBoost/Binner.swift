import Foundation

/// Quantile bin-edge computation (CPU, parallel over features). The actual
/// binning of the full matrix happens on the GPU (`bin_data` kernel); this
/// only sorts a subsample per feature to pick the cut points, exactly like
/// LightGBM's bin mapper.
///
/// Bin layout: numBins - 1 data bins + 1 reserved missing bin, so there are
/// numBins - 2 edges per feature. NaNs are excluded from the quantiles.
/// Categorical features need no edges (category id maps straight to a bin).
enum Binner {
    /// Returns cols * (numBins - 2) upper edges, feature-major.
    static func computeEdges(featureMajor X: [Float], rows: Int, cols: Int,
                             numBins: Int, categorical: Set<Int>,
                             sampleCap: Int = 200_000) -> [Float] {
        let dataBins = numBins - 1
        let numEdges = dataBins - 1
        let step = max(1, rows / sampleCap)
        var edges = [Float](repeating: 0, count: cols * numEdges)
        edges.withUnsafeMutableBufferPointer { ep in
            X.withUnsafeBufferPointer { xp in
                DispatchQueue.concurrentPerform(iterations: cols) { f in
                    if categorical.contains(f) { return }
                    var sample = [Float]()
                    sample.reserveCapacity(rows / step + 1)
                    var i = 0
                    while i < rows {
                        let v = xp[f * rows + i]
                        if !v.isNaN { sample.append(v) }
                        i += step
                    }
                    guard !sample.isEmpty else { return }   // all-NaN column
                    sample.sort()
                    let m = sample.count
                    for k in 0..<numEdges {
                        let idx = min(m - 1, ((k + 1) * m) / dataBins)
                        ep[f * numEdges + k] = sample[idx]
                    }
                }
            }
        }
        return edges
    }
}
