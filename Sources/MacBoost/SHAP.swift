import Foundation

// TreeSHAP (Lundberg & Lee): exact Shapley value feature contributions for
// tree ensembles, using each node's training cover as the background
// distribution. CPU implementation, parallel over rows.

private struct PathElement {
    var featureIndex: Int
    var zeroFraction: Double
    var oneFraction: Double
    var pweight: Double
}

extension MacBooster {

    /// Per-feature SHAP contributions plus a trailing expected-value column
    /// per class: shape rows x (cols + 1) for single-output models, rows x
    /// numClasses*(cols + 1) for multiclass (LightGBM's pred_contrib
    /// layout). Rows (per class block) sum to the RAW score.
    public func predictContributions(featureMajor X: [Float], rows: Int,
                                     cols: Int) -> [Float] {
        precondition(cols == numFeatures)
        let K = trainedNumClasses
        let width = cols + 1
        var out = [Float](repeating: 0, count: rows * K * width)
        let localTrees = trees
        let dataBins = trainedNumBins - 1

        // Expected value per class (base + cover-weighted tree means).
        var bias = [Double](repeating: 0, count: K)
        for k in 0..<K { bias[k] = Double(baseScores[k]) }
        for (t, tree) in localTrees.enumerated() {
            var mean = 0.0
            var stack = [0]
            let total = Double(tree.cover.first ?? 0)
            guard total > 0 else { continue }
            while let n = stack.popLast() {
                if tree.feature[n] >= 0 { stack += [2 * n + 1, 2 * n + 2] }
                else { mean += Double(tree.leaf[n]) * Double(tree.cover[n]) / total }
            }
            bias[t % K] += mean
        }

        let chunk = 512
        let nChunks = (rows + chunk - 1) / chunk
        X.withUnsafeBufferPointer { xp in
            out.withUnsafeMutableBufferPointer { op in
                DispatchQueue.concurrentPerform(iterations: nChunks) { c in
                    var phi = [Double](repeating: 0, count: width)
                    for i in (c * chunk)..<min(rows, (c + 1) * chunk) {
                        for k in 0..<K {
                            for j in 0..<width { phi[j] = 0 }
                            var t = k
                            while t < localTrees.count {
                                let tree = localTrees[t]
                                if (tree.cover.first ?? 0) > 0 {
                                    shapRecurse(tree: tree, node: 0,
                                                path: [], parentZero: 1, parentOne: 1,
                                                parentFeature: -1,
                                                x: xp, sampleIndex: i, rows: rows,
                                                dataBins: dataBins, phi: &phi)
                                }
                                t += K
                            }
                            let base = i * K * width + k * width
                            for f in 0..<cols { op[base + f] = Float(phi[f]) }
                            op[base + cols] = Float(bias[k])
                        }
                    }
                }
            }
        }
        return out
    }

    private func goesLeftRaw(_ tree: Tree, _ node: Int, _ v: Float,
                             dataBins: Int) -> Bool {
        if UInt32(tree.flags[node]) & flagCategorical != 0 {
            var cat = dataBins
            if v.isFinite {
                let r = v.rounded(.toNearestOrEven)
                if r >= 0 && Int(r) < dataBins { cat = Int(r) }
            }
            return (tree.catMask[node * 8 + cat / 32] >> (cat % 32)) & 1 != 0
        }
        if v.isNaN { return UInt32(tree.flags[node]) & flagDefaultLeft != 0 }
        return v <= tree.threshold[node]
    }

    private func shapRecurse(tree: Tree, node: Int, path: [PathElement],
                             parentZero: Double, parentOne: Double,
                             parentFeature: Int,
                             x: UnsafeBufferPointer<Float>, sampleIndex: Int,
                             rows: Int, dataBins: Int,
                             phi: inout [Double]) {
        var m = extendPath(path, zeroFraction: parentZero,
                           oneFraction: parentOne, featureIndex: parentFeature)
        if tree.feature[node] < 0 {
            // Leaf: attribute along every feature on the path.
            let v = Double(tree.leaf[node])
            for i in 1..<m.count {
                let w = unwoundPathSum(m, i)
                phi[m[i].featureIndex] += w * (m[i].oneFraction - m[i].zeroFraction) * v
            }
            return
        }
        let f = Int(tree.feature[node])
        let hot = goesLeftRaw(tree, node, x[f * rows + sampleIndex], dataBins: dataBins)
            ? 2 * node + 1 : 2 * node + 2
        let cold = hot == 2 * node + 1 ? 2 * node + 2 : 2 * node + 1
        let coverNode = Double(tree.cover[node])
        let hotFrac = coverNode > 0 ? Double(tree.cover[hot]) / coverNode : 0
        let coldFrac = coverNode > 0 ? Double(tree.cover[cold]) / coverNode : 0

        var incomingZero = 1.0, incomingOne = 1.0
        if let k = m.firstIndex(where: { $0.featureIndex == f }) {
            incomingZero = m[k].zeroFraction
            incomingOne = m[k].oneFraction
            m = unwindPath(m, k)
        }
        shapRecurse(tree: tree, node: hot, path: m,
                    parentZero: incomingZero * hotFrac, parentOne: incomingOne,
                    parentFeature: f, x: x, sampleIndex: sampleIndex,
                    rows: rows, dataBins: dataBins, phi: &phi)
        shapRecurse(tree: tree, node: cold, path: m,
                    parentZero: incomingZero * coldFrac, parentOne: 0,
                    parentFeature: f, x: x, sampleIndex: sampleIndex,
                    rows: rows, dataBins: dataBins, phi: &phi)
    }

    private func extendPath(_ path: [PathElement], zeroFraction: Double,
                            oneFraction: Double, featureIndex: Int) -> [PathElement] {
        var m = path
        let l = m.count
        m.append(PathElement(featureIndex: featureIndex, zeroFraction: zeroFraction,
                             oneFraction: oneFraction, pweight: l == 0 ? 1 : 0))
        if l > 0 {
            for i in stride(from: l - 1, through: 0, by: -1) {
                m[i + 1].pweight += oneFraction * m[i].pweight * Double(i + 1) / Double(l + 1)
                m[i].pweight = zeroFraction * m[i].pweight * Double(l - i) / Double(l + 1)
            }
        }
        return m
    }

    private func unwindPath(_ path: [PathElement], _ index: Int) -> [PathElement] {
        var m = path
        let l = m.count - 1
        let one = m[index].oneFraction
        let zero = m[index].zeroFraction
        var next = m[l].pweight
        if one != 0 {
            for i in stride(from: l - 1, through: 0, by: -1) {
                let tmp = m[i].pweight
                m[i].pweight = next * Double(l + 1) / (Double(i + 1) * one)
                next = tmp - m[i].pweight * zero * Double(l - i) / Double(l + 1)
            }
        } else {
            for i in stride(from: l - 1, through: 0, by: -1) {
                m[i].pweight = m[i].pweight * Double(l + 1) / (zero * Double(l - i))
            }
        }
        for i in index..<l {
            m[i].featureIndex = m[i + 1].featureIndex
            m[i].zeroFraction = m[i + 1].zeroFraction
            m[i].oneFraction = m[i + 1].oneFraction
        }
        m.removeLast()
        return m
    }

    private func unwoundPathSum(_ path: [PathElement], _ index: Int) -> Double {
        let l = path.count - 1
        let one = path[index].oneFraction
        let zero = path[index].zeroFraction
        var next = path[l].pweight
        var total = 0.0
        if one != 0 {
            for i in stride(from: l - 1, through: 0, by: -1) {
                let tmp = next / (Double(i + 1) * one)
                total += tmp
                next = path[i].pweight - tmp * zero * Double(l - i)
            }
        } else if zero != 0 {
            for i in stride(from: l - 1, through: 0, by: -1) {
                total += path[i].pweight / (zero * Double(l - i))
            }
        }
        return total * Double(l + 1)
    }
}
