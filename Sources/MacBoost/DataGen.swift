import Foundation

/// Deterministic, parallel synthetic data. Friedman #1-style regression:
/// y = 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4 + noise,
/// remaining features are irrelevant noise columns.
public struct SplitMix64 {
    var state: UInt64
    public init(seed: UInt64) { state = seed }

    public mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var z = state
        z = (z ^ (z >> 30)) &* 0xBF58476D1CE4E5B9
        z = (z ^ (z >> 27)) &* 0x94D049BB133111EB
        return z ^ (z >> 31)
    }

    public mutating func uniform() -> Float {
        Float(next() >> 40) * (1.0 / Float(1 << 24))
    }

    public mutating func normal() -> Float {
        let u1 = max(uniform(), 1e-7), u2 = uniform()
        return (-2 * log(u1)).squareRoot() * cos(2 * .pi * u2)
    }
}

public enum DataGen {
    /// Friedman #1 numerics plus categorical columns appended after them.
    /// `catCards[j]` is the cardinality of categorical column numericCols+j;
    /// the first two categorical columns carry real signal (a per-category
    /// offset in [-3, 3] derived from `effectSeed`, so train/valid sets built
    /// with different sample seeds share the same ground truth). NaNs are
    /// injected into the 5 informative numeric columns at `missingFrac`
    /// AFTER y is computed (the value exists, the observation is missing).
    public static func friedmanMixed(rows: Int, numericCols: Int, catCards: [Int],
                                     seed: UInt64, effectSeed: UInt64,
                                     missingFrac: Float = 0, noise: Float = 1.0)
        -> ([Float], [Float]) {
        var (X, y) = friedman(rows: rows, cols: numericCols, seed: seed, noise: noise)
        X.append(contentsOf: [Float](repeating: 0, count: rows * catCards.count))

        for (j, card) in catCards.enumerated() {
            var effects = [Float](repeating: 0, count: card)
            var erng = SplitMix64(seed: effectSeed &+ UInt64(j) &* 0x9E3779B97F4A7C15)
            for c in 0..<card { effects[c] = erng.uniform() * 6 - 3 }
            let informative = j < 2
            var rng = SplitMix64(seed: seed &+ 0xC0FFEE &+ UInt64(j))
            let base = (numericCols + j) * rows
            for i in 0..<rows {
                let c = Int(rng.next() % UInt64(card))
                X[base + i] = Float(c)
                if informative { y[i] += effects[c] }
            }
        }
        if missingFrac > 0 {
            var rng = SplitMix64(seed: seed ^ 0x4D495353)   // "MISS"
            for f in 0..<min(5, numericCols) {
                for i in 0..<rows where rng.uniform() < missingFrac {
                    X[f * rows + i] = .nan
                }
            }
        }
        return (X, y)
    }

    /// Returns (X feature-major, y).
    public static func friedman(rows: Int, cols: Int, seed: UInt64, noise: Float = 1.0) -> ([Float], [Float]) {
        precondition(cols >= 5)
        var X = [Float](repeating: 0, count: rows * cols)
        X.withUnsafeMutableBufferPointer { xp in
            DispatchQueue.concurrentPerform(iterations: cols) { f in
                var rng = SplitMix64(seed: seed &+ UInt64(f) &* 0x9E3779B97F4A7C15)
                for i in 0..<rows { xp[f * rows + i] = rng.uniform() }
            }
        }
        var y = [Float](repeating: 0, count: rows)
        let chunk = 65_536
        let nChunks = (rows + chunk - 1) / chunk
        y.withUnsafeMutableBufferPointer { yp in
            X.withUnsafeBufferPointer { xp in
                DispatchQueue.concurrentPerform(iterations: nChunks) { c in
                    var rng = SplitMix64(seed: (seed ^ 0xDEADBEEF) &+ UInt64(c) &* 0x9E3779B97F4A7C15)
                    for i in (c * chunk)..<min(rows, (c + 1) * chunk) {
                        let x0 = xp[i], x1 = xp[rows + i], x2 = xp[2 * rows + i]
                        let x3 = xp[3 * rows + i], x4 = xp[4 * rows + i]
                        yp[i] = 10 * sin(.pi * x0 * x1) + 20 * (x2 - 0.5) * (x2 - 0.5)
                              + 10 * x3 + 5 * x4 + noise * rng.normal()
                    }
                }
            }
        }
        return (X, y)
    }
}
