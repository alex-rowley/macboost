import Foundation
import Metal

// TreeSHAP (Lundberg & Lee): exact Shapley value feature contributions for
// tree ensembles, using each node's training cover as the background
// distribution. Large batches run on the GPU via path decomposition (one
// thread per row x leaf path, GPUTreeSHAP-style); small ones on a CPU
// implementation parallel over rows. Both produce the same values (the GPU
// accumulates in float32, so equality is to float precision, not bitwise).

private struct PathElement {
    var featureIndex: Int
    var zeroFraction: Double
    var oneFraction: Double
    var pweight: Double
}

// Must match the MSL ShapElement layout (5 x 4 bytes).
private struct ShapElementHost {
    var feature: UInt32
    var zeroFraction: Float
    var threshold: Float
    var flags: UInt32          // FLAG_* bits; bit 8 = path continues LEFT
    var nodeGlobal: UInt32     // tree*nodesPerTree + node (catMask lookup)
}

private struct ShapParamsHost {
    var totalRows: UInt32; var rowCount: UInt32; var rowOffset: UInt32
    var numPaths: UInt32; var numClasses: UInt32; var numFeatures: UInt32
    var dataBins: UInt32
}

private let shapChildLeft: UInt32 = 0x100
private let shapMaxPathLen = 13        // must match SHAP_MAX_PATH in MSL
private let shapGPUMinRows = 256

extension MacBooster {

    /// Per-feature SHAP contributions plus a trailing expected-value column
    /// per class: shape rows x (cols + 1) for single-output models, rows x
    /// numClasses*(cols + 1) for multiclass (LightGBM's pred_contrib
    /// layout). Rows (per class block) sum to the RAW score.
    public func predictContributions(featureMajor X: [Float], rows: Int,
                                     cols: Int) -> [Float] {
        precondition(cols == numFeatures)
        if rows >= shapGPUMinRows, !trees.isEmpty,
           let out = predictContributionsGPU(featureMajor: X, rows: rows,
                                             cols: cols) {
            return out
        }
        return predictContributionsCPU(featureMajor: X, rows: rows, cols: cols)
    }

    /// Expected raw score per class: base score + cover-weighted tree means.
    /// This is the trailing bias column of the contributions output.
    private func shapBias() -> [Double] {
        let K = trainedNumClasses
        var bias = [Double](repeating: 0, count: K)
        for k in 0..<K { bias[k] = Double(baseScores[k]) }
        for (t, tree) in trees.enumerated() {
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
        return bias
    }

    // MARK: - GPU path

    /// Decompose the forest into root-to-leaf paths for the gpu_treeshap
    /// kernel. Returns nil when the model can't run on the GPU path (no
    /// covers stored, or a path deeper than the kernel's register arrays).
    private func extractShapPaths(nodesPerTree: Int)
        -> (elements: [ShapElementHost], starts: [UInt32],
            leaves: [Float], classes: [UInt32])? {
        let K = trainedNumClasses
        var elements: [ShapElementHost] = []
        var starts: [UInt32] = [0]
        var leaves: [Float] = []
        var classes: [UInt32] = []
        for (t, tree) in trees.enumerated() {
            guard tree.cover.count == tree.feature.count,
                  tree.cover[0] > 0 else { return nil }
            var stack: [(node: Int, path: [ShapElementHost])] = [(0, [])]
            while let (n, path) = stack.popLast() {
                if tree.feature[n] < 0 {
                    if path.isEmpty { continue }     // root-is-leaf: bias only
                    elements.append(contentsOf: path)
                    starts.append(UInt32(elements.count))
                    leaves.append(tree.leaf[n])
                    classes.append(UInt32(t % K))
                    continue
                }
                if path.count + 1 > shapMaxPathLen - 1 { return nil }
                let coverN = tree.cover[n]
                for (child, isLeft) in [(2 * n + 1, true), (2 * n + 2, false)] {
                    let el = ShapElementHost(
                        feature: UInt32(tree.feature[n]),
                        zeroFraction: coverN > 0 ? tree.cover[child] / coverN : 0,
                        threshold: tree.threshold[n],
                        flags: UInt32(tree.flags[n]) | (isLeft ? shapChildLeft : 0),
                        nodeGlobal: UInt32(t * nodesPerTree + n))
                    stack.append((child, path + [el]))
                }
            }
        }
        return elements.isEmpty ? nil : (elements, starts, leaves, classes)
    }

    private func predictContributionsGPU(featureMajor X: [Float], rows: Int,
                                         cols: Int) -> [Float]? {
        let forest = forestBuffers()
        guard let paths = extractShapPaths(nodesPerTree: forest.nodesPerTree)
        else { return nil }
        let K = trainedNumClasses
        let width = cols + 1
        let numPaths = paths.leaves.count
        let outLength = rows * K * width * 4

        let xBuf = engine.makeBuffer(X)
        let elBuf = engine.makeBuffer(paths.elements)
        let stBuf = engine.makeBuffer(paths.starts)
        let lfBuf = engine.makeBuffer(paths.leaves)
        let clBuf = engine.makeBuffer(paths.classes)
        let outBuf = engine.makeBuffer(length: outLength)

        // Bound each dispatch to ~2^28 row-path threads so a huge request
        // becomes several short command buffers instead of one long one.
        let chunk = max(1, min(rows, (1 << 28) / max(numPaths, 1)))
        var offset = 0
        while offset < rows {
            let count = min(chunk, rows - offset)
            let cb = engine.queue.makeCommandBuffer()!
            if offset == 0 { engine.fillZero(cb, outBuf, length: outLength) }
            engine.dispatch(cb, "gpu_treeshap",
                            buffers: [xBuf, elBuf, stBuf, lfBuf, clBuf,
                                      forest.masks, outBuf],
                            params: ShapParamsHost(
                                totalRows: UInt32(rows), rowCount: UInt32(count),
                                rowOffset: UInt32(offset),
                                numPaths: UInt32(numPaths),
                                numClasses: UInt32(K), numFeatures: UInt32(cols),
                                dataBins: UInt32(trainedNumBins - 1)),
                            grid: MTLSize(width: count, height: numPaths, depth: 1),
                            threadgroup: MTLSize(width: 64, height: 4, depth: 1))
            cb.commit()
            cb.waitUntilCompleted()
            offset += count
        }

        var out = [Float](UnsafeBufferPointer(
            start: outBuf.contents().bindMemory(to: Float.self,
                                                capacity: rows * K * width),
            count: rows * K * width))
        let bias = shapBias()
        for i in 0..<rows {
            for k in 0..<K { out[i * K * width + k * width + cols] = Float(bias[k]) }
        }
        return out
    }

    // MARK: - CPU path

    func predictContributionsCPU(featureMajor X: [Float], rows: Int,
                                 cols: Int) -> [Float] {
        let K = trainedNumClasses
        let width = cols + 1
        var out = [Float](repeating: 0, count: rows * K * width)
        let localTrees = trees
        let dataBins = trainedNumBins - 1
        let bias = shapBias()

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
