import Foundation
import Metal
import Accelerate

public enum Objective: UInt32 {
    case regression = 0       // squared error (l2)
    case binaryLogistic = 1
    case mae = 2              // absolute error (l1)
    case huber = 3            // huber loss, delta = alpha
    case quantile = 4         // pinball loss at quantile alpha
    case poisson = 5          // poisson deviance, log link
    case tweedie = 6          // tweedie deviance, log link, power in (1,2)
    case multiclass = 7       // softmax; set numClasses

    var usesLogLink: Bool { self == .poisson || self == .tweedie }
}

public struct BoosterParams {
    public var numTrees = 100
    public var maxDepth = 6            // splits happen at depths 0..maxDepth-1
    public var learningRate: Float = 0.1
    public var lambda: Float = 1.0     // L2 regularisation on leaf values
    public var minChildHess: Float = 1.0
    public var minSplitGain: Float = 0.0
    public var numBins = 256           // includes 1 reserved missing bin; <= 256
    public var objective = Objective.regression
    public var categoricalFeatures: Set<Int> = []   // column indices; values 0..<numBins-1
    public var catSmooth: Float = 10   // denominator smoothing in category ordering
    /// GOSS (gradient-based one-side sampling): train each tree on the
    /// top `gossTopRate` fraction of samples by |gradient| plus a uniform
    /// `gossOtherRate` sample of the rest (gradients amplified by
    /// (1-a)/b), like LightGBM's data_sample_strategy=goss.
    public var goss = false
    public var gossTopRate: Float = 0.2
    public var gossOtherRate: Float = 0.1
    /// Huber delta / quantile level (LightGBM's shared `alpha`).
    public var alpha: Float = 0.9
    /// Tweedie variance power, in (1, 2).
    public var tweedieVariancePower: Float = 1.5
    /// Multiplier on positive-class gradients for binaryLogistic.
    public var scalePosWeight: Float = 1.0
    /// Uniform per-tree row sampling fraction (bagging). 1 = off.
    public var subsample: Float = 1.0
    /// Per-tree feature sampling fraction (colsample_bytree). 1 = off.
    public var featureFraction: Float = 1.0
    /// Per-feature monotone constraints (+1 increasing, -1 decreasing, 0
    /// none), LightGBM's basic enforcement mode. Count must equal cols.
    public var monotoneConstraints: [Int]?
    /// Validation metric; .auto picks the objective's default.
    public var metric: EvalMetric = .auto
    /// Number of classes for the multiclass objective (labels 0..<K).
    public var numClasses: Int = 1
    /// When set, splits only consider these feature indices (the model
    /// still accepts full-width X at predict time). Set automatically by
    /// feature selection; composes with featureFraction sampling.
    public var allowedFeatures: Set<Int>?
    public init() {}
}

public enum EvalMetric: String, CaseIterable {
    case auto, rmse, mae, logloss, auc, pinball, poisson, tweedie
    case multiLogloss = "multi_logloss"
}

public struct EvalSet {
    public let featureMajor: [Float]
    public let rows: Int
    public let labels: [Float]
    public init(featureMajor: [Float], rows: Int, labels: [Float]) {
        self.featureMajor = featureMajor
        self.rows = rows
        self.labels = labels
    }
}

public struct FitTimings {
    public var binning = 0.0          // edge computation + GPU bin (train + valid)
    public var gpuLevels = 0.0        // scatter + histograms + split search
    public var cpuDecide = 0.0        // host-side split decisions
    public var finalize = 0.0         // partition + prediction/valid updates
    public var total = 0.0
}

/// A trained tree in implicit-heap layout. feature[n] < 0 marks a leaf.
/// flags bit0: missing values go left; bit1: categorical split (membership
/// of the sample's bin in catMask decides left). catMask holds 8 UInt32
/// words per node (256-bin bitset); empty when the model has no
/// categorical features.
public struct Tree: Codable {
    public var feature: [Int32]
    public var threshold: [Float]     // numeric splits: go left if x <= t
    public var leaf: [Float]          // learning rate already applied
    public var flags: [UInt8]
    public var catMask: [UInt32]
    public var gain: [Float]          // split gain per node (0 at leaves)
    public var cover: [Float]         // training sample count per node

    public init(feature: [Int32], threshold: [Float], leaf: [Float],
                flags: [UInt8], catMask: [UInt32],
                gain: [Float] = [], cover: [Float] = []) {
        self.feature = feature; self.threshold = threshold; self.leaf = leaf
        self.flags = flags; self.catMask = catMask
        self.gain = gain; self.cover = cover
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        feature = try c.decode([Int32].self, forKey: .feature)
        threshold = try c.decode([Float].self, forKey: .threshold)
        leaf = try c.decode([Float].self, forKey: .leaf)
        flags = try c.decode([UInt8].self, forKey: .flags)
        catMask = try c.decode([UInt32].self, forKey: .catMask)
        gain = try c.decodeIfPresent([Float].self, forKey: .gain) ?? []
        cover = try c.decodeIfPresent([Float].self, forKey: .cover) ?? []
    }
}

let flagDefaultLeft: UInt32 = 1
let flagCategorical: UInt32 = 2

// Param structs mirroring the MSL structs (all 4-byte fields, no padding).
struct BinParams { var numSamples: UInt32; var numFeatures: UInt32; var numBins: UInt32 }
private struct GradParams {
    var numSamples: UInt32; var objective: UInt32
    var alpha: Float; var aux: Float; var hasWeights: UInt32
}
private struct RouteParams {
    var levelStart: UInt32; var numLevel: UInt32; var numSamples: UInt32
    var numBins: UInt32; var terminal: UInt32; var apply: UInt32; var predOffset: UInt32
}
private struct GradMCParams {
    var numSamples: UInt32; var numClasses: UInt32; var classIndex: UInt32; var hasWeights: UInt32
}
private struct GossParams {
    var numSamples: UInt32; var treeSeed: UInt32; var otherProb: Float; var weight: Float
}
private struct GossFinalizeParams { var samplesPerGroup: UInt32; var numTiles: UInt32 }
private struct QuantParams { var numSamples: UInt32 }
private struct HistParams {
    var numSamples: UInt32; var numFeatures: UInt32; var numBins: UInt32
    var numNodes: UInt32; var samplesPerGroup: UInt32
}
private struct SubParams { var sliceLen: UInt32 }
private struct InitParams { var rows: UInt32; var samplesPerGroup: UInt32; var numTiles: UInt32 }
private struct DecideParams {
    var levelStart: UInt32; var numLevel: UInt32; var numFeatures: UInt32; var numBins: UInt32
    var numTiles: UInt32; var samplesPerGroup: UInt32; var isLastLevel: UInt32
    var lambda: Float; var learningRate: Float; var minSplitGain: Float; var catSmooth: Float
}
private struct FinalParams {
    var lastStart: UInt32; var numLast: UInt32; var lambda: Float; var learningRate: Float
}
private struct SplitParams {
    var numFeatures: UInt32; var numBins: UInt32; var numNodes: UInt32
    var lambda: Float; var minChildHess: Float; var minSplitGain: Float; var catSmooth: Float
}
private struct SplitResult {
    var gain: Float; var bin: UInt32; var gl: Float; var hl: Float; var flags: UInt32
}
private struct NodeSplit { var feature: Int32; var bin: UInt32; var flags: UInt32 }
private struct PredictParams {
    var numSamples: UInt32; var maxDepth: UInt32; var numBins: UInt32; var predOffset: UInt32
}
private struct ForestParams {
    var numSamples: UInt32; var numFeatures: UInt32; var numTrees: UInt32
    var nodesPerTree: UInt32; var dataBins: UInt32; var numClasses: UInt32; var transform: UInt32
}

private let histChannels = 3
private let tgSize = 256
private let tileF = 8
// Hess+count share one 32-bit accumulator: count in 13 bits, 7-bit-quantised
// hess sums in 19 bits. samplesPerGroup must satisfy spg <= 8191 (count) and
// spg * 127 < 2^19 (hess) — 4096 fits both with headroom.
private let samplesPerGroup = 4096

public final class MacBooster {
    public let params: BoosterParams
    public private(set) var trees: [Tree] = []
    public private(set) var baseScore: Float = 0
    public private(set) var numFeatures = 0
    /// 1-based iteration with the best validation metric (set when a valid
    /// set is passed to fit). With earlyStoppingRounds > 0 the model is
    /// truncated to exactly this many trees.
    public private(set) var bestIteration: Int?
    /// Validation metric per completed iteration (RMSE or logloss).
    public private(set) var evalHistory: [Float] = []
    /// Optional column names, persisted in the model file (set by callers
    /// such as the CLI so predict can map columns by name).
    public var featureNames: [String]?
    /// Data-quality warnings from the last fit (e.g. suspected target
    /// leakage). Also emitted through the progress callback.
    public private(set) var fitWarnings: [String] = []

    // Effective training configuration: normally params', but a
    // BinnedDataset overrides them (its binning is baked in).
    var trainedNumBins: Int          // internal: SHAP needs the bin layout
    private var trainedCategorical: Set<Int>
    /// Per-class base scores (single-element for non-multiclass).
    public private(set) var baseScores: [Float] = [0]
    /// Number of classes the model was trained with (1 unless multiclass).
    public internal(set) var trainedNumClasses = 1

    let engine: MetalEngine

    public convenience init(params: BoosterParams) throws {
        try self.init(params: params, engine: nil)
    }

    init(params: BoosterParams, engine: MetalEngine?) throws {
        guard params.numBins >= 4 && params.numBins <= 256 else {
            throw MacBoostError.invalidInput("numBins must be in 4...256, got \(params.numBins)")
        }
        guard params.maxDepth >= 1 && params.maxDepth <= 12 else {
            throw MacBoostError.invalidInput("maxDepth must be in 1...12, got \(params.maxDepth)")
        }
        guard params.numTrees >= 1 else {
            throw MacBoostError.invalidInput("numTrees must be >= 1, got \(params.numTrees)")
        }
        guard params.learningRate.isFinite && params.learningRate > 0 else {
            throw MacBoostError.invalidInput("learningRate must be positive and finite")
        }
        if params.goss {
            guard params.gossTopRate > 0 && params.gossOtherRate > 0
                    && params.gossTopRate + params.gossOtherRate <= 1 else {
                throw MacBoostError.invalidInput(
                    "GOSS rates must be positive with topRate + otherRate <= 1")
            }
            guard params.subsample >= 1 else {
                throw MacBoostError.invalidInput("goss and subsample are mutually exclusive")
            }
        }
        if params.objective == .quantile {
            guard params.alpha > 0 && params.alpha < 1 else {
                throw MacBoostError.invalidInput("quantile alpha must be in (0, 1)")
            }
        }
        if params.objective == .tweedie {
            guard params.tweedieVariancePower > 1 && params.tweedieVariancePower < 2 else {
                throw MacBoostError.invalidInput("tweedieVariancePower must be in (1, 2)")
            }
        }
        guard params.subsample > 0 && params.subsample <= 1 else {
            throw MacBoostError.invalidInput("subsample must be in (0, 1]")
        }
        guard params.featureFraction > 0 && params.featureFraction <= 1 else {
            throw MacBoostError.invalidInput("featureFraction must be in (0, 1]")
        }
        guard params.scalePosWeight > 0 else {
            throw MacBoostError.invalidInput("scalePosWeight must be positive")
        }
        if let mc = params.monotoneConstraints,
           !mc.allSatisfy({ (-1...1).contains($0) }) {
            throw MacBoostError.invalidInput("monotone constraints must be -1, 0 or +1")
        }
        if params.metric == .auc && params.objective != .binaryLogistic {
            throw MacBoostError.invalidInput("auc metric requires the binary objective")
        }
        if params.objective == .multiclass {
            guard (3...128).contains(params.numClasses) else {
                throw MacBoostError.invalidInput(
                    "multiclass requires numClasses in 3...128 (use binaryLogistic for 2)")
            }
            guard !params.goss else {
                throw MacBoostError.invalidInput("GOSS is not supported with multiclass yet")
            }
        }
        self.params = params
        self.trainedNumBins = params.numBins
        self.trainedCategorical = params.categoricalFeatures
        self.engine = try engine ?? MetalEngine()
    }

    public var deviceName: String { engine.device.name }

    // MARK: - Training

    /// Train on a feature-major float matrix (X[f * rows + i]). NaN marks a
    /// missing value. Columns listed in params.categoricalFeatures must hold
    /// integer category ids in 0..<numBins-1 (or NaN).
    @discardableResult
    public func fit(featureMajor X: [Float], rows: Int, cols: Int, labels: [Float],
                    weights: [Float]? = nil, valid: EvalSet? = nil,
                    earlyStoppingRounds: Int = 0, evalEvery: Int = 0,
                    initModel: MacBooster? = nil,
                    progress: ((String) -> Void)? = nil) throws -> FitTimings {
        try fitImpl(X: X, prebinned: nil, rows: rows, cols: cols, labels: labels,
                    weights: weights, valid: valid,
                    earlyStoppingRounds: earlyStoppingRounds,
                    evalEvery: evalEvery, initModel: initModel, progress: progress)
    }

    /// Train from a preprocessed BinnedDataset (see makeDataset / .mbds
    /// files): skips parsing and binning entirely. The dataset's bin count
    /// and categorical layout are used, not the booster params'.
    @discardableResult
    public func fit(dataset: BinnedDataset, weights: [Float]? = nil,
                    valid: EvalSet? = nil, earlyStoppingRounds: Int = 0,
                    evalEvery: Int = 0, progress: ((String) -> Void)? = nil) throws -> FitTimings {
        if featureNames == nil { featureNames = dataset.featureNames }
        return try fitImpl(X: nil, prebinned: dataset, rows: dataset.rows,
                           cols: dataset.cols, labels: dataset.labels,
                           weights: weights, valid: valid,
                           earlyStoppingRounds: earlyStoppingRounds,
                           evalEvery: evalEvery, initModel: nil, progress: progress)
    }

    /// Preprocess (validate, compute quantile edges, GPU-bin) a raw matrix
    /// into a reusable BinnedDataset.
    public func makeDataset(featureMajor X: [Float], rows: Int, cols: Int,
                            labels: [Float], featureNames: [String]? = nil) throws -> BinnedDataset {
        guard rows > 0 && cols > 0, X.count == rows * cols, labels.count == rows else {
            throw MacBoostError.invalidInput(
                "dataset shapes inconsistent (rows=\(rows), cols=\(cols), X=\(X.count), y=\(labels.count))")
        }
        for (i, v) in labels.enumerated() where !v.isFinite {
            throw MacBoostError.invalidInput("label at row \(i) is \(v); labels must be finite")
        }
        try validateCategoricals(X, rows: rows, cols: cols)
        let nBins = params.numBins
        var featFlags = [UInt8](repeating: 0, count: cols)
        for f in params.categoricalFeatures { featFlags[f] = 1 }
        let edges = Binner.computeEdges(featureMajor: X, rows: rows, cols: cols,
                                        numBins: nBins,
                                        categorical: params.categoricalFeatures)
        let numTiles = (cols + tileF - 1) / tileF
        let binsBuf = engine.makeBuffer(length: rows * numTiles * tileF)
        let cb = engine.queue.makeCommandBuffer()!
        engine.dispatch(cb, "bin_data",
                        buffers: [engine.makeBuffer(X), engine.makeBuffer(edges),
                                  binsBuf, engine.makeBuffer(featFlags)],
                        params: BinParams(numSamples: UInt32(rows),
                                          numFeatures: UInt32(cols),
                                          numBins: UInt32(nBins)),
                        grid: MTLSize(width: rows, height: cols, depth: 1),
                        threadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        cb.commit()
        cb.waitUntilCompleted()
        let bins = [UInt8](UnsafeBufferPointer(
            start: binsBuf.contents().bindMemory(to: UInt8.self, capacity: binsBuf.length),
            count: rows * numTiles * tileF))
        return BinnedDataset(rows: rows, cols: cols, numBins: nBins,
                             categorical: params.categoricalFeatures,
                             featureNames: featureNames, labels: labels,
                             bins: bins, edges: edges)
    }

    /// GPU-resident pre-binned input (feature selection's shadow matrix):
    /// tiled bin bytes that never exist host-side.
    struct GPUBinned {
        let bins: MTLBuffer          // tiled layout, `cols` features wide
        let edges: [Float]
        let categorical: Set<Int>
        let numBins: Int
    }

    func fitImpl(X: [Float]?, prebinned: BinnedDataset?,
                 gpuBinned: GPUBinned? = nil, rows: Int, cols: Int,
                         labels: [Float], weights: [Float]?, valid: EvalSet?,
                         earlyStoppingRounds: Int, evalEvery: Int,
                         initModel: MacBooster?,
                         progress: ((String) -> Void)?) throws -> FitTimings {
        guard rows > 0 && cols > 0 else {
            throw MacBoostError.invalidInput("training data is empty (rows=\(rows), cols=\(cols))")
        }
        if let X {
            guard X.count == rows * cols else {
                throw MacBoostError.invalidInput(
                    "X has \(X.count) values, expected rows*cols = \(rows * cols)")
            }
        }
        guard labels.count == rows else {
            throw MacBoostError.invalidInput("labels has \(labels.count) rows, X has \(rows)")
        }
        try validateLabels(labels)
        if let w = weights {
            guard w.count == rows else {
                throw MacBoostError.invalidInput("weights has \(w.count) rows, X has \(rows)")
            }
            for (i, v) in w.enumerated() where !(v.isFinite && v >= 0) {
                throw MacBoostError.invalidInput("weight at row \(i) is \(v); weights must be finite and >= 0")
            }
        }
        if let v = valid {
            guard v.featureMajor.count == v.rows * cols else {
                throw MacBoostError.invalidInput(
                    "valid X has \(v.featureMajor.count) values, expected \(v.rows * cols) " +
                    "(same feature count as training data)")
            }
            guard v.labels.count == v.rows else {
                throw MacBoostError.invalidInput("valid labels/rows mismatch")
            }
            try validateLabels(v.labels)
        }
        if let mc = params.monotoneConstraints, mc.count != cols {
            throw MacBoostError.invalidInput(
                "monotoneConstraints has \(mc.count) entries, X has \(cols) features")
        }
        if let af = params.allowedFeatures {
            guard !af.isEmpty, af.allSatisfy({ $0 >= 0 && $0 < cols }) else {
                throw MacBoostError.invalidInput(
                    "allowedFeatures must be non-empty indices within 0..<\(cols)")
            }
        }
        if let im = initModel {
            guard X != nil else {
                throw MacBoostError.invalidInput(
                    "initModel requires raw features (not a prebinned dataset)")
            }
            guard im.numFeatures == cols else {
                throw MacBoostError.invalidInput(
                    "initModel expects \(im.numFeatures) features, X has \(cols)")
            }
            guard im.params.objective == params.objective else {
                throw MacBoostError.invalidInput("initModel objective differs")
            }
        }
        fitWarnings = []
        if let X {
            checkLabelLeakage(X, rows: rows, cols: cols, labels: labels, progress: progress)
        }
        numFeatures = cols
        var timings = FitTimings()
        let tTotal = Clock.now()

        // The dataset (when given) dictates binning and categorical layout.
        let categorical = gpuBinned?.categorical ?? prebinned?.categorical
            ?? params.categoricalFeatures
        let nBins = gpuBinned?.numBins ?? prebinned?.numBins ?? params.numBins
        trainedNumBins = nBins
        trainedCategorical = categorical
        let dataBins = nBins - 1
        let maxDepth = params.maxDepth
        let maxBuildNodes = 1 << (maxDepth - 1)          // widest level that gets histograms
        let totalNodes = (1 << (maxDepth + 1)) - 1
        let lr = params.learningRate
        let lambda = params.lambda
        let hasCats = !categorical.isEmpty

        if let X { try validateCategoricals(X, rows: rows, cols: cols) }
        if let v = valid {
            try validateCategoricals(v.featureMajor, rows: v.rows, cols: cols)
        }

        var featFlags = [UInt8](repeating: 0, count: cols)
        for f in categorical { featFlags[f] = 1 }
        let featFlagsBuf = engine.makeBuffer(featFlags)
        let numTiles = (cols + tileF - 1) / tileF
        // Grad AND hess quantisation scales are GPU-reduced per tree
        // (dynamic), so unbounded hessians (poisson/tweedie), sample
        // weights and GOSS amplification all keep full 7-bit precision.
        let gossWeight: Float = params.goss
            ? (1 - params.gossTopRate) / params.gossOtherRate : 1

        // --- Binning (train + optional valid, same edges) -----------------
        // With a prebinned dataset the train matrix is already binned; only
        // an optional raw valid set still goes through bin_data.
        var t0 = Clock.now()
        let edges: [Float]
        let binsBuf: MTLBuffer
        let validBinsBuf = valid.map { engine.makeBuffer(length: $0.rows * numTiles * tileF) }
        if let gb = gpuBinned {
            edges = gb.edges
            binsBuf = gb.bins
        } else if let ds = prebinned {
            edges = ds.edges
            binsBuf = engine.makeBuffer(ds.bins)
        } else {
            edges = Binner.computeEdges(featureMajor: X!, rows: rows, cols: cols,
                                        numBins: nBins, categorical: categorical)
            binsBuf = engine.makeBuffer(length: rows * numTiles * tileF)
        }
        do {
            let cb = engine.queue.makeCommandBuffer()!
            var needsCommit = false
            if prebinned == nil && gpuBinned == nil {
                let edgesBuf = engine.makeBuffer(edges)
                let xBuf = engine.makeBuffer(X!)
                engine.dispatch(cb, "bin_data",
                                buffers: [xBuf, edgesBuf, binsBuf, featFlagsBuf],
                                params: BinParams(numSamples: UInt32(rows),
                                                  numFeatures: UInt32(cols),
                                                  numBins: UInt32(nBins)),
                                grid: MTLSize(width: rows, height: cols, depth: 1),
                                threadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                needsCommit = true
            }
            if let v = valid, let vb = validBinsBuf {
                let edgesBuf = engine.makeBuffer(edges)
                let vxBuf = engine.makeBuffer(v.featureMajor)
                engine.dispatch(cb, "bin_data",
                                buffers: [vxBuf, edgesBuf, vb, featFlagsBuf],
                                params: BinParams(numSamples: UInt32(v.rows),
                                                  numFeatures: UInt32(cols),
                                                  numBins: UInt32(nBins)),
                                grid: MTLSize(width: v.rows, height: cols, depth: 1),
                                threadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
                needsCommit = true
            }
            if needsCommit {
                cb.commit()
                cb.waitUntilCompleted()
            }
        } // raw X + edges buffers released here
        timings.binning = Clock.since(t0)

        // --- Persistent buffers -------------------------------------------
        let ghBuf = engine.makeBuffer(length: rows * 8)          // float2 (grad, hess)
        let ghqBuf = engine.makeBuffer(length: rows * 4)         // packed 16-bit qg | 7-bit qh
        let maxGHBuf = engine.makeBuffer(length: 16)   // f32-bits atomic max |g|, max h
        let weightsBuf = weights.map { engine.makeBuffer($0) } ?? engine.makeBuffer(length: 16)
        let hasWeights: UInt32 = weights != nil ? 1 : 0
        let K = params.objective == .multiclass ? params.numClasses : 1
        trainedNumClasses = K
        let predsBuf = engine.makeBuffer(length: rows * K * 4)
        let labelsBuf = engine.makeBuffer(labels)
        // Node-grouped sample order per level: level 0 reads the identity
        // permutation; deeper levels ping-pong between two buffers, filled by
        // route_samples with work proportional to still-active rows.
        let orderIdentBuf = engine.makeBuffer(length: rows * 4)
        let orderABuf = engine.makeBuffer(length: rows * 4)
        let orderBBuf = engine.makeBuffer(length: rows * 4)
        // GOSS compacts the kept sample ids into orderB as level 0's input
        // (it is only read by level 0/1 work, before route(1) reuses it).
        // Like LightGBM, sampling starts after 1/learningRate warm-up trees.
        let gossWarmup = params.goss ? Int((1.0 / params.learningRate).rounded(.up)) : 0
        var gossActive = false
        var treeSampled = false
        func orderBuf(_ d: Int) -> MTLBuffer {
            d == 0 ? (treeSampled ? orderBBuf : orderIdentBuf)
                   : (d % 2 == 1 ? orderABuf : orderBBuf)
        }
        let gossBucketsBuf = engine.makeBuffer(length: 1024 * 4)
        let gossThresholdBuf = engine.makeBuffer(length: 16)
        let gossCursorBuf = engine.makeBuffer(length: 16)
        let maxWGHBuf = engine.makeBuffer(length: 16)    // weighted maxima (GOSS/bagging)
        var monotone = [Int8](repeating: 0, count: cols)
        if let mc = params.monotoneConstraints {
            for f in 0..<cols { monotone[f] = Int8(mc[f]) }
        }
        let monotoneBuf = engine.makeBuffer(monotone)
        let boundsBuf = engine.makeBuffer(length: totalNodes * 8)   // float2 leaf bounds
        // Leaf renewal (LightGBM RenewTreeOutput): L1 and quantile leaves are
        // recomputed as residual quantiles after the tree structure is
        // decided. Forces one sync per tree.
        let leafRenewal = params.objective == .mae || params.objective == .quantile
        let renewalQ = params.objective == .mae ? 0.5 : params.alpha
        let leafIdxBuf = leafRenewal ? engine.makeBuffer(length: rows * 4) : engine.makeBuffer(length: 16)
        let boundsPtr = boundsBuf.contents().bindMemory(to: SIMD2<Float>.self, capacity: totalNodes)
        // Per-level metadata, all GPU-written (decide_splits / init_tree):
        // segment starts+cursors+build counts per node, live-sample totals,
        // and the two indirect dispatch argument sets.
        let segStartL = (0..<maxDepth).map { engine.makeBuffer(length: (1 << $0) * 4) }
        let segCursorL = (0..<maxDepth).map { engine.makeBuffer(length: (1 << $0) * 4) }
        let buildCountL = (0..<maxDepth).map { engine.makeBuffer(length: (1 << $0) * 4) }
        let activeTotalL = (0..<maxDepth).map { _ in engine.makeBuffer(length: 16) }
        let histArgsL = (0..<maxDepth).map { _ in engine.makeBuffer(length: 16) }
        let routeArgsL = (0..<maxDepth).map { _ in engine.makeBuffer(length: 16) }
        let histLen = maxBuildNodes * cols * nBins * histChannels * 4
        var histCurBuf = engine.makeBuffer(length: histLen)
        var histPrevBuf = engine.makeBuffer(length: histLen)
        let deriveMapBuf = engine.makeBuffer(length: maxBuildNodes * 3 * 4)
        let splitResBuf = engine.makeBuffer(length: maxBuildNodes * cols * MemoryLayout<SplitResult>.stride)
        let nodeSplitsBuf = engine.makeBuffer(length: totalNodes * MemoryLayout<NodeSplit>.stride)
        let leafValuesBuf = engine.makeBuffer(length: totalNodes * 4)
        let statsBuf = engine.makeBuffer(length: totalNodes * 16)     // (G,H,count) float4
        let numDeriveBuf = engine.makeBuffer(length: 16)
        let validPredsBuf = valid.map { engine.makeBuffer(length: $0.rows * K * 4) }

        // Per-tree output buffers are cycled so trees pipeline: the CPU
        // reads tree t's structure back while the GPU already runs later
        // trees. Multiclass batches a whole K-tree round into one command
        // buffer, so it needs K buffer sets per round in flight (2 rounds
        // without a valid set, 1 with). Single-output keeps depth 4 (1
        // with a valid set, which forces a per-tree metric sync).
        let pipelineDepth = K > 1 ? (valid == nil ? 2 * K : K)
                                  : (valid == nil ? 4 : 1)
        let nodeSplitsBufs = (0..<pipelineDepth).map { _ in
            engine.makeBuffer(length: totalNodes * MemoryLayout<NodeSplit>.stride)
        }
        let leafValuesBufs = (0..<pipelineDepth).map { _ in
            engine.makeBuffer(length: totalNodes * 4)
        }
        let catMaskBufs = (0..<pipelineDepth).map { _ in
            engine.makeBuffer(length: totalNodes * 8 * 4)
        }
        let gainsBufs = (0..<pipelineDepth).map { _ in
            engine.makeBuffer(length: totalNodes * 4)
        }
        let coverBufs = (0..<pipelineDepth).map { _ in
            engine.makeBuffer(length: totalNodes * 4)
        }
        // Per-tree feature masks (colsample_bytree and/or allowedFeatures);
        // all-ones when both are off.
        let colsampling = params.featureFraction < 1
        var maskBase = [UInt8](repeating: 1, count: cols)
        if let af = params.allowedFeatures {
            for f in 0..<cols { maskBase[f] = af.contains(f) ? 1 : 0 }
        }
        let featMaskBufs = (0..<(colsampling ? pipelineDepth : 1)).map { _ in
            engine.makeBuffer(maskBase)
        }
        let predsPtr = predsBuf.contents().bindMemory(to: Float.self, capacity: rows * K)

        // --- Startup info preamble (zero cost when progress is nil) --------
        if let progress {
            let numCats = params.categoricalFeatures.count
            let numEdges = dataBins - 1
            var totalBins = 0
            for f in 0..<cols {
                if params.categoricalFeatures.contains(f) { totalBins += dataBins; continue }
                var distinct = 1
                for k in 1..<numEdges where edges[f * numEdges + k] != edges[f * numEdges + k - 1] {
                    distinct += 1
                }
                totalBins += min(distinct + 1, dataBins)
            }
            var bufferBytes = [binsBuf, ghBuf, ghqBuf, predsBuf, labelsBuf, orderIdentBuf,
                               orderABuf, orderBBuf, histCurBuf, histPrevBuf, splitResBuf, statsBuf]
                .reduce(0) { $0 + $1.length }
            bufferBytes += (nodeSplitsBufs + leafValuesBufs + catMaskBufs).reduce(0) { $0 + $1.length }
            if let vb = validBinsBuf { bufferBytes += vb.length }
            if let vp = validPredsBuf { bufferBytes += vp.length }

            let validInfo = valid.map { ", valid: \($0.rows) rows" } ?? ""
            progress("[MacBoost] [Info] train: \(rows) rows, \(cols) features " +
                     "(\(cols - numCats) numeric, \(numCats) categorical)\(validInfo)")
            progress(String(format: "[MacBoost] [Info] %@, total bins %d, binning %.3fs, " +
                            "GPU buffers %.1f MB unified",
                            engine.device.name, totalBins, timings.binning,
                            Double(bufferBytes) / 1_048_576))
            progress(String(format: "[MacBoost] [Info] objective=%@ trees=%d depth=%d " +
                            "lr=%g lambda=%g minChildHess=%g histBins=%d",
                            params.objective == .regression ? "regression" : "binary",
                            params.numTrees, maxDepth, params.learningRate,
                            lambda, params.minChildHess, nBins))
        }

        // --- Base score ----------------------------------------------------
        func weightedMean() -> Float {
            guard let w = weights else {
                var m: Float = 0
                vDSP_meanv(labels, 1, &m, vDSP_Length(rows))
                return m
            }
            var num: Float = 0, den: Float = 0
            vDSP_dotpr(labels, 1, w, 1, &num, vDSP_Length(rows))
            vDSP_sve(w, 1, &den, vDSP_Length(rows))
            return den > 0 ? num / den : 0
        }
        func subsampleQuantile(_ q: Float) -> Float {
            let step = max(1, rows / 200_000)
            var sample = [Float]()
            var i = 0
            while i < rows { sample.append(labels[i]); i += step }
            sample.sort()
            let idx = min(sample.count - 1, Int(Float(sample.count) * q))
            return sample[idx]
        }
        switch params.objective {
        case .regression, .huber:
            baseScore = weightedMean()
        case .mae:
            baseScore = subsampleQuantile(0.5)
        case .quantile:
            baseScore = subsampleQuantile(params.alpha)
        case .binaryLogistic:
            // Weighted positive rate including scale_pos_weight; fast vDSP
            // path for the common unweighted case.
            let p: Float
            if weights == nil && params.scalePosWeight == 1 {
                var mean: Float = 0
                vDSP_meanv(labels, 1, &mean, vDSP_Length(rows))
                p = min(max(mean, 1e-6), 1 - 1e-6)
            } else {
                var pos: Double = 0, tot: Double = 0
                let spw = Double(params.scalePosWeight)
                labels.withUnsafeBufferPointer { lp in
                    if let w = weights {
                        w.withUnsafeBufferPointer { wp in
                            for i in 0..<rows {
                                let wi = Double(wp[i])
                                if lp[i] > 0.5 { pos += wi * spw; tot += wi * spw }
                                else { tot += wi }
                            }
                        }
                    } else {
                        for i in 0..<rows {
                            if lp[i] > 0.5 { pos += spw; tot += spw } else { tot += 1 }
                        }
                    }
                }
                p = min(max(Float(pos / max(tot, 1e-12)), 1e-6), 1 - 1e-6)
            }
            baseScore = log(p / (1 - p))
        case .poisson, .tweedie:
            let m = weightedMean()
            guard m > 0 else {
                throw MacBoostError.invalidInput(
                    "\(params.objective) requires a positive label mean")
            }
            baseScore = log(m)
        case .multiclass:
            baseScore = 0
        }
        if K > 1 {
            var counts = [Double](repeating: 1, count: K)   // +1 smoothing
            for i in 0..<rows { counts[Int(labels[i])] += Double(weights?[i] ?? 1) }
            let total = counts.reduce(0, +)
            baseScores = counts.map { Float(log($0 / total)) }
        } else {
            baseScores = [baseScore]
        }
        if let im = initModel, let X {
            // Warm start: running predictions begin at the init model's raw
            // scores; its trees become the ensemble prefix.
            baseScore = im.baseScore
            let seed = im.predictRawScores(featureMajor: X, rows: rows, cols: cols)
            seed.withUnsafeBufferPointer { predsPtr.update(from: $0.baseAddress!, count: rows) }
            if let v = valid, let vp = validPredsBuf {
                let vseed = im.predictRawScores(featureMajor: v.featureMajor,
                                                rows: v.rows, cols: cols)
                vseed.withUnsafeBufferPointer {
                    vp.contents().bindMemory(to: Float.self, capacity: v.rows)
                        .update(from: $0.baseAddress!, count: v.rows)
                }
            }
        } else {
            for k in 0..<K {
                var base = baseScores[k]
                vDSP_vfill(&base, predsPtr + k * rows, 1, vDSP_Length(rows))
                if let v = valid, let vp = validPredsBuf {
                    vDSP_vfill(&base, vp.contents().bindMemory(
                        to: Float.self, capacity: v.rows * K) + k * v.rows,
                               1, vDSP_Length(v.rows))
                }
            }
        }

        let rowGrid = MTLSize(width: rows, height: 1, depth: 1)
        let tg1D = MTLSize(width: tgSize, height: 1, depth: 1)

        // Identity permutation for level 0 (built once, read-only afterwards).
        do {
            let cb = engine.queue.makeCommandBuffer()!
            engine.dispatch(cb, "iota", buffers: [orderIdentBuf],
                            params: UInt32(rows), grid: rowGrid, threadgroup: tg1D)
            cb.commit()
        }

        trees = initModel?.trees ?? []
        let initTreeCount = trees.count
        let totalTrees = params.numTrees * K
        trees.reserveCapacity(initTreeCount + totalTrees)
        evalHistory.removeAll()
        bestIteration = nil
        var bestMetric = metricHigherIsBetter ? -Float.infinity : Float.infinity
        var bestIter = 0
        var staleRounds = 0

        // In-flight trees: committed command buffers whose structures the
        // host has not read back yet.
        var pending: [(index: Int, cb: MTLCommandBuffer)] = []
        func drainPending(keep: Int) {
            while pending.count > keep {
                let (idx, pcb) = pending.removeFirst()
                pcb.waitUntilCompleted()
                let k = idx % pipelineDepth
                let nsPtr = nodeSplitsBufs[k].contents()
                    .bindMemory(to: NodeSplit.self, capacity: totalNodes)
                let lvPtr = leafValuesBufs[k].contents()
                    .bindMemory(to: Float.self, capacity: totalNodes)
                let cmPtr = catMaskBufs[k].contents()
                    .bindMemory(to: UInt32.self, capacity: totalNodes * 8)
                let gnPtr = gainsBufs[k].contents()
                    .bindMemory(to: Float.self, capacity: totalNodes)
                let cvPtr = coverBufs[k].contents()
                    .bindMemory(to: Float.self, capacity: totalNodes)
                // Reachable nodes only — deeper buffer entries are stale
                // leftovers from earlier trees.
                var tree = Tree(feature: [Int32](repeating: -1, count: totalNodes),
                                threshold: [Float](repeating: 0, count: totalNodes),
                                leaf: [Float](repeating: 0, count: totalNodes),
                                flags: [UInt8](repeating: 0, count: totalNodes),
                                catMask: hasCats ? [UInt32](repeating: 0, count: totalNodes * 8) : [],
                                gain: [Float](repeating: 0, count: totalNodes),
                                cover: [Float](repeating: 0, count: totalNodes))
                var stack = [0]
                while let n = stack.popLast() {
                    let s = nsPtr[n]
                    tree.cover[n] = cvPtr[n]
                    if s.feature >= 0 {
                        tree.feature[n] = s.feature
                        tree.flags[n] = UInt8(s.flags)
                        tree.gain[n] = gnPtr[n]
                        if s.flags & flagCategorical != 0 {
                            for w in 0..<8 { tree.catMask[n * 8 + w] = cmPtr[n * 8 + w] }
                        } else {
                            tree.threshold[n] = edges[Int(s.feature) * (dataBins - 1) + Int(s.bin)]
                        }
                        stack.append(2 * n + 1)
                        stack.append(2 * n + 2)
                    } else {
                        tree.leaf[n] = lvPtr[n]
                    }
                }
                trees.append(tree)
            }
        }

        var roundCB: MTLCommandBuffer?
        var roundTreeIds: [Int] = []
        for t in 0..<totalTrees {
            let classIdx = t % K
            let k = t % pipelineDepth
            let nodeSplitsBuf = nodeSplitsBufs[k]
            let leafValuesBuf = leafValuesBufs[k]
            let catMaskBuf = catMaskBufs[k]
            let gainsBuf = gainsBufs[k]
            let coverBuf = coverBufs[k]
            let featMaskBuf = featMaskBufs[colsampling ? k : 0]
            // Encode the ENTIRE tree as one command buffer: gradients,
            // quantisation, then per level scatter -> histograms ->
            // subtraction -> split search -> GPU split decisions, and finally
            // partition + leaf application. No CPU round-trips inside a tree;
            // level sizes the CPU cannot know are handled by decide_splits
            // writing the next level's indirect dispatch arguments.
            t0 = Clock.now()
            // Multiclass: all K trees of a round share one command buffer
            // (commits are costly at small row counts). Consecutive
            // dispatches share compute encoders too, broken only where an
            // indirect dispatch needs its arguments written by an earlier
            // encoder (init/goss -> level 0, decide(d) -> level d+1).
            let cb: MTLCommandBuffer
            if K > 1, let rc = roundCB {
                cb = rc
            } else {
                cb = engine.queue.makeCommandBuffer()!
                roundCB = cb
            }
            var e = engine.beginCompute(cb)
            e.zero(maxGHBuf, length: 8)
            e.zero(statsBuf, length: totalNodes * 16)
            if K > 1 {
                e.dispatch("compute_gradients_multiclass",
                                buffers: [predsBuf, labelsBuf, weightsBuf, ghBuf, maxGHBuf],
                                params: GradMCParams(numSamples: UInt32(rows),
                                                     numClasses: UInt32(K),
                                                     classIndex: UInt32(classIdx),
                                                     hasWeights: hasWeights),
                                grid: rowGrid, threadgroup: tg1D)
            } else {
                let aux: Float
                switch params.objective {
                case .binaryLogistic: aux = params.scalePosWeight
                case .tweedie: aux = params.tweedieVariancePower
                default: aux = 0
                }
                e.dispatch("compute_gradients",
                                buffers: [predsBuf, labelsBuf, weightsBuf, ghBuf, maxGHBuf],
                                params: GradParams(numSamples: UInt32(rows),
                                                   objective: params.objective.rawValue,
                                                   alpha: params.alpha, aux: aux,
                                                   hasWeights: hasWeights),
                                grid: rowGrid, threadgroup: tg1D)
            }
            e.dispatch("init_tree",
                            buffers: [segStartL[0], buildCountL[0], activeTotalL[0],
                                      numDeriveBuf, histArgsL[0], routeArgsL[0], boundsBuf],
                            params: InitParams(rows: UInt32(rows),
                                               samplesPerGroup: UInt32(samplesPerGroup),
                                               numTiles: UInt32(numTiles)),
                            grid: MTLSize(width: 1, height: 1, depth: 1),
                            threadgroup: MTLSize(width: 1, height: 1, depth: 1))
            gossActive = params.goss && t >= gossWarmup
            let bagging = params.subsample < 1
            let samplingActive = gossActive || bagging
            treeSampled = samplingActive
            if colsampling {
                let maskPtr = featMaskBuf.contents().bindMemory(to: UInt8.self, capacity: cols)
                var rng = SplitMix64(seed: 0xC0150001 &+ UInt64(t) &* 0x9E3779B97F4A7C15)
                var any = false
                for f in 0..<cols {
                    let keep: UInt8 = maskBase[f] == 1
                        && rng.uniform() < params.featureFraction ? 1 : 0
                    maskPtr[f] = keep
                    any = any || keep == 1
                }
                if !any {
                    let candidates = (0..<cols).filter { maskBase[$0] == 1 }
                    maskPtr[candidates[Int(rng.next() % UInt64(candidates.count))]] = 1
                }
            }
            if samplingActive {
                e.zero(gossCursorBuf, length: 4)
                e.zero(maxWGHBuf, length: 8)
                if gossActive {
                    e.zero(gossBucketsBuf, length: 1024 * 4)
                    e.dispatch("goss_grad_hist",
                                    buffers: [ghBuf, maxGHBuf, gossBucketsBuf],
                                    params: UInt32(rows), grid: rowGrid, threadgroup: tg1D)
                    e.dispatch("goss_threshold",
                                    buffers: [gossBucketsBuf, maxGHBuf, gossThresholdBuf],
                                    params: UInt32(Float(rows) * params.gossTopRate),
                                    grid: MTLSize(width: 1, height: 1, depth: 1),
                                    threadgroup: MTLSize(width: 1, height: 1, depth: 1))
                } else {
                    // Bagging: topCount=0 puts the threshold at max|g|, so
                    // (near-)everything goes through the hashed uniform
                    // keep-test at rate `subsample`, all with weight 1.
                    e.dispatch("goss_threshold",
                                    buffers: [gossBucketsBuf, maxGHBuf, gossThresholdBuf],
                                    params: UInt32(0),
                                    grid: MTLSize(width: 1, height: 1, depth: 1),
                                    threadgroup: MTLSize(width: 1, height: 1, depth: 1))
                }
                let otherProb = gossActive
                    ? params.gossOtherRate / (1 - params.gossTopRate)
                    : params.subsample
                e.dispatch("goss_select",
                                buffers: [ghBuf, gossThresholdBuf, orderBBuf,
                                          gossCursorBuf, maxWGHBuf],
                                params: GossParams(numSamples: UInt32(rows),
                                                   treeSeed: UInt32(t / K &+ 1),
                                                   otherProb: otherProb,
                                                   weight: gossActive ? gossWeight : 1),
                                grid: rowGrid, threadgroup: tg1D)
                e.dispatch("goss_finalize",
                                buffers: [gossCursorBuf, buildCountL[0], activeTotalL[0],
                                          histArgsL[0], routeArgsL[0]],
                                params: GossFinalizeParams(
                                    samplesPerGroup: UInt32(samplesPerGroup),
                                    numTiles: UInt32(numTiles)),
                                grid: MTLSize(width: 1, height: 1, depth: 1),
                                threadgroup: MTLSize(width: 1, height: 1, depth: 1))
            }
            e.dispatch("quantize_gradients",
                            buffers: [ghBuf, samplingActive ? maxWGHBuf : maxGHBuf, ghqBuf],
                            params: QuantParams(numSamples: UInt32(rows)),
                            grid: rowGrid, threadgroup: tg1D)

            func encodeRoute(_ d: Int, terminal: Bool) {
                e.dispatchIndirect("route_samples",
                                        buffers: [orderBuf(d),
                                                  terminal ? orderBuf(d) : orderBuf(d + 1),
                                                  segStartL[d],
                                                  terminal ? segCursorL[d] : segCursorL[d + 1],
                                                  activeTotalL[d], nodeSplitsBuf, leafValuesBuf,
                                                  catMaskBuf, binsBuf, predsBuf],
                                        params: RouteParams(levelStart: UInt32((1 << d) - 1),
                                                            numLevel: UInt32(1 << d),
                                                            numSamples: UInt32(rows),
                                                            numBins: UInt32(nBins),
                                                            terminal: terminal ? 1 : 0,
                                                            apply: (treeSampled || leafRenewal) ? 0 : 1,
                                                            predOffset: UInt32(classIdx * rows)),
                                        indirect: routeArgsL[d], threadgroup: tg1D)
            }

            for d in 0..<maxDepth {
                let levelStart = (1 << d) - 1
                let numLevel = 1 << d
                e.end()
                e = engine.beginCompute(cb)
                if d > 0 {
                    encodeRoute(d - 1, terminal: false)
                    swap(&histCurBuf, &histPrevBuf)
                }
                e.zero(histCurBuf, length: numLevel * cols * nBins * histChannels * 4)
                e.dispatchIndirect("build_histograms",
                                        buffers: [binsBuf, ghqBuf, orderBuf(d),
                                                  segStartL[d], buildCountL[d], histCurBuf,
                                                  samplingActive ? maxWGHBuf : maxGHBuf],
                                        params: HistParams(numSamples: UInt32(rows),
                                                           numFeatures: UInt32(cols),
                                                           numBins: UInt32(nBins),
                                                           numNodes: UInt32(numLevel),
                                                           samplesPerGroup: UInt32(samplesPerGroup)),
                                        indirect: histArgsL[d], threadgroup: tg1D)
                if d > 0 {
                    let sliceLen = cols * nBins * histChannels
                    e.dispatch("subtract_histograms",
                                    buffers: [histPrevBuf, histCurBuf, deriveMapBuf, numDeriveBuf],
                                    params: SubParams(sliceLen: UInt32(sliceLen)),
                                    grid: MTLSize(width: sliceLen, height: numLevel / 2, depth: 1),
                                    threadgroup: tg1D)
                }
                e.dispatch("find_splits",
                                buffers: [histCurBuf, featFlagsBuf, featMaskBuf,
                                          monotoneBuf, splitResBuf],
                                params: SplitParams(numFeatures: UInt32(cols),
                                                    numBins: UInt32(nBins),
                                                    numNodes: UInt32(numLevel),
                                                    lambda: lambda,
                                                    minChildHess: params.minChildHess,
                                                    minSplitGain: params.minSplitGain,
                                                    catSmooth: params.catSmooth),
                                grid: MTLSize(width: cols, height: numLevel, depth: 1),
                                threadgroup: tg1D)
                let nextLevel = min(d + 1, maxDepth - 1)
                e.dispatch("decide_splits",
                                buffers: [splitResBuf, histCurBuf, nodeSplitsBuf, leafValuesBuf,
                                          catMaskBuf, statsBuf,
                                          segStartL[nextLevel], segCursorL[nextLevel],
                                          buildCountL[nextLevel], deriveMapBuf, numDeriveBuf,
                                          activeTotalL[nextLevel], histArgsL[nextLevel],
                                          routeArgsL[nextLevel], gainsBuf,
                                          boundsBuf, monotoneBuf],
                                params: DecideParams(levelStart: UInt32(levelStart),
                                                     numLevel: UInt32(numLevel),
                                                     numFeatures: UInt32(cols),
                                                     numBins: UInt32(nBins),
                                                     numTiles: UInt32(numTiles),
                                                     samplesPerGroup: UInt32(samplesPerGroup),
                                                     isLastLevel: d == maxDepth - 1 ? 1 : 0,
                                                     lambda: lambda,
                                                     learningRate: lr,
                                                     minSplitGain: params.minSplitGain,
                                                     catSmooth: params.catSmooth),
                                grid: MTLSize(width: tgSize, height: 1, depth: 1),
                                threadgroup: tg1D)
            }

            // Deepest level: mark leaves, then the terminal route applies
            // every remaining sample's leaf value to its prediction.
            e.end()
            e = engine.beginCompute(cb)
            e.dispatch("final_leaves",
                            buffers: [nodeSplitsBuf, leafValuesBuf, statsBuf, boundsBuf],
                            params: FinalParams(lastStart: UInt32((1 << maxDepth) - 1),
                                                numLast: UInt32(1 << maxDepth),
                                                lambda: lambda, learningRate: lr),
                            grid: MTLSize(width: 1 << maxDepth, height: 1, depth: 1),
                            threadgroup: tg1D)
            e.dispatch("copy_cover",
                            buffers: [statsBuf, coverBuf],
                            params: UInt32(totalNodes),
                            grid: MTLSize(width: totalNodes, height: 1, depth: 1),
                            threadgroup: tg1D)
            encodeRoute(maxDepth - 1, terminal: true)
            if leafRenewal {
                e.dispatch("assign_leaves",
                                buffers: [binsBuf, nodeSplitsBuf, catMaskBuf, leafIdxBuf],
                                params: PredictParams(numSamples: UInt32(rows),
                                                      maxDepth: UInt32(maxDepth),
                                                      numBins: UInt32(nBins),
                                                      predOffset: 0),
                                grid: rowGrid, threadgroup: tg1D)
            }
            if samplingActive && !leafRenewal {
                // Sampling means routed leaf application only covered kept
                // rows; walk the finished tree for EVERY row instead.
                e.dispatch("predict_tree_binned",
                                buffers: [binsBuf, predsBuf, nodeSplitsBuf,
                                          leafValuesBuf, catMaskBuf],
                                params: PredictParams(numSamples: UInt32(rows),
                                                      maxDepth: UInt32(maxDepth),
                                                      numBins: UInt32(nBins),
                                                      predOffset: UInt32(classIdx * rows)),
                                grid: rowGrid, threadgroup: tg1D)
            }
            if !leafRenewal, let v = valid, let vb = validBinsBuf, let vp = validPredsBuf {
                e.dispatch("predict_tree_binned",
                                buffers: [vb, vp, nodeSplitsBuf, leafValuesBuf, catMaskBuf],
                                params: PredictParams(numSamples: UInt32(v.rows),
                                                      maxDepth: UInt32(maxDepth),
                                                      numBins: UInt32(nBins),
                                                      predOffset: UInt32(classIdx * v.rows)),
                                grid: MTLSize(width: v.rows, height: 1, depth: 1),
                                threadgroup: tg1D)
            }
            e.end()
            let roundEnd = K == 1 || classIdx == K - 1 || t + 1 == totalTrees
            if roundEnd {
                cb.commit()
                roundCB = nil
            }
            if leafRenewal {
                cb.waitUntilCompleted()
                // Per-leaf residual quantiles from a stride subsample.
                let leafIdxPtr = leafIdxBuf.contents().bindMemory(to: Int32.self, capacity: rows)
                let lvPtr = leafValuesBuf.contents().bindMemory(to: Float.self, capacity: totalNodes)
                let step = max(1, rows / 262_144)
                var buckets = [Int: [Float]]()
                var i = 0
                while i < rows {
                    let leaf = Int(leafIdxPtr[i])
                    buckets[leaf, default: []].append(labels[i] - predsPtr[i])
                    i += step
                }
                for (leaf, var res) in buckets where res.count >= 4 {
                    res.sort()
                    let q = res[min(res.count - 1, Int(Float(res.count) * renewalQ))]
                    let bb = boundsPtr[leaf]
                    lvPtr[leaf] = lr * min(max(q, bb.x), bb.y)
                }
                // Apply renewed leaves to train preds (+ valid) in a second
                // command buffer.
                let cb2 = engine.queue.makeCommandBuffer()!
                engine.dispatch(cb2, "predict_tree_binned",
                                buffers: [binsBuf, predsBuf, nodeSplitsBuf,
                                          leafValuesBuf, catMaskBuf],
                                params: PredictParams(numSamples: UInt32(rows),
                                                      maxDepth: UInt32(maxDepth),
                                                      numBins: UInt32(nBins),
                                                      predOffset: 0),
                                grid: rowGrid, threadgroup: tg1D)
                if let v = valid, let vb = validBinsBuf, let vp = validPredsBuf {
                    engine.dispatch(cb2, "predict_tree_binned",
                                    buffers: [vb, vp, nodeSplitsBuf, leafValuesBuf, catMaskBuf],
                                    params: PredictParams(numSamples: UInt32(v.rows),
                                                          maxDepth: UInt32(maxDepth),
                                                          numBins: UInt32(nBins),
                                                          predOffset: 0),
                                    grid: MTLSize(width: v.rows, height: 1, depth: 1),
                                    threadgroup: tg1D)
                }
                cb2.commit()
                pending.append((t, cb2))
            } else if K == 1 {
                pending.append((t, cb))
            } else {
                roundTreeIds.append(t)
                if roundEnd {
                    for tt in roundTreeIds { pending.append((tt, cb)) }
                    roundTreeIds.removeAll()
                }
            }
            timings.gpuLevels += Clock.since(t0)

            let isLast = t + 1 == totalTrees
            let roundDone = (t + 1) % K == 0
            let round = (t + 1) / K
            let logNow = evalEvery > 0 && roundDone && (round % evalEvery == 0 || isLast)
            let needAll = (valid != nil && roundDone) || logNow || isLast
            t0 = Clock.now()
            drainPending(keep: needAll ? 0 : pipelineDepth - 1)
            timings.cpuDecide += Clock.since(t0)

            t0 = Clock.now()
            var line: String?
            if logNow {
                let m = K > 1
                    ? multiclassLogloss(predsPtr: predsPtr, labels: labels, rows: rows, K: K)
                    : trainMetric(predsPtr: predsPtr, labels: labels, rows: rows)
                line = String(format: "[%4d] train-%@: %.5f", round, metricName, m)
            }
            if let v = valid, let vp = validPredsBuf, roundDone {
                let vpPtr = vp.contents().bindMemory(to: Float.self, capacity: v.rows * K)
                let vm = K > 1
                    ? multiclassLogloss(predsPtr: vpPtr, labels: v.labels, rows: v.rows, K: K)
                    : evalMetric(predsPtr: vpPtr, labels: v.labels, rows: v.rows)
                evalHistory.append(vm)
                if metricHigherIsBetter ? vm > bestMetric : vm < bestMetric {
                    bestMetric = vm; bestIter = round; staleRounds = 0
                } else {
                    staleRounds += 1
                }
                if logNow { line! += String(format: "   valid-%@: %.5f", metricName, vm) }
                if earlyStoppingRounds > 0 && staleRounds >= earlyStoppingRounds {
                    if let l = line { progress?(l) }
                    progress?(String(format: "early stopping at iteration %d (best: %d, valid-%@ %.5f)",
                                     round, bestIter, metricName, bestMetric))
                    timings.finalize += Clock.since(t0)
                    break
                }
            }
            if let l = line { progress?(l) }
            timings.finalize += Clock.since(t0)
        }

        if valid != nil {
            bestIteration = bestIter
            if earlyStoppingRounds > 0 && trees.count > initTreeCount + bestIter * K {
                trees.removeSubrange((initTreeCount + bestIter * K)...)
            }
        }

        forestCache = nil
        timings.total = Clock.since(tTotal)
        return timings
    }

    private var effectiveMetric: EvalMetric {
        guard params.metric == .auto else { return params.metric }
        switch params.objective {
        case .regression, .huber: return .rmse
        case .mae: return .mae
        case .quantile: return .pinball
        case .binaryLogistic: return .logloss
        case .poisson: return .poisson
        case .tweedie: return .tweedie
        case .multiclass: return .multiLogloss
        }
    }

    private var metricHigherIsBetter: Bool { effectiveMetric == .auc }

    private var metricName: String { effectiveMetric.rawValue }

    /// Labels must be finite; binary logistic requires 0/1; poisson and
    /// tweedie require non-negative labels; multiclass requires class ids.
    /// Fast path: one vectorised scan, with a per-row pass only to localise
    /// an error.
    func validateLabels(_ labels: [Float]) throws {
        let logistic = params.objective == .binaryLogistic
        let nonNegative = params.objective.usesLogLink
        let multiclass = params.objective == .multiclass
        let K = Float(params.numClasses)
        var suspicious = false
        labels.withUnsafeBufferPointer { lp in
            var ok = true
            for i in 0..<lp.count {
                let v = lp[i]
                ok = ok && v.isFinite
                if logistic { ok = ok && (v == 0 || v == 1) }
                if nonNegative { ok = ok && v >= 0 }
                if multiclass {
                    ok = ok && v >= 0 && v < K && v.rounded(.toNearestOrEven) == v
                }
            }
            suspicious = !ok
        }
        guard suspicious else { return }
        for (i, v) in labels.enumerated() {
            if !v.isFinite {
                throw MacBoostError.invalidInput("label at row \(i) is \(v); labels must be finite")
            }
            if logistic && v != 0 && v != 1 {
                throw MacBoostError.invalidInput(
                    "label at row \(i) is \(v); binary objective requires labels in {0, 1}")
            }
            if nonNegative && v < 0 {
                throw MacBoostError.invalidInput(
                    "label at row \(i) is \(v); \(params.objective) requires labels >= 0")
            }
            if multiclass {
                let r = v.rounded(.toNearestOrEven)
                if r != v || r < 0 || Int(r) >= params.numClasses {
                    throw MacBoostError.invalidInput(
                        "label at row \(i) is \(v); multiclass requires integer labels " +
                        "in 0..<\(params.numClasses)")
                }
            }
        }
    }

    /// Cheap guard against the most common data bug: a feature column that
    /// IS the label (or a prior prediction of it) left in X. Checks
    /// |correlation| on a small subsample — costs microseconds, catches the
    /// mistake neither LightGBM nor XGBoost warns about.
    private func checkLabelLeakage(_ X: [Float], rows: Int, cols: Int, labels: [Float],
                                   progress: ((String) -> Void)?) {
        let step = max(1, rows / 4096)
        var idx = [Int]()
        var i = 0
        while i < rows { idx.append(i); i += step }
        guard idx.count >= 50 else { return }

        var ym = 0.0
        for i in idx { ym += Double(labels[i]) }
        ym /= Double(idx.count)

        for f in 0..<cols {
            var sxy = 0.0, sxx = 0.0, syy = 0.0, xm = 0.0
            var n = 0
            for i in idx where X[f * rows + i].isFinite {
                xm += Double(X[f * rows + i]); n += 1
            }
            guard n >= 50 else { continue }
            xm /= Double(n)
            for i in idx {
                let x = Double(X[f * rows + i])
                guard x.isFinite else { continue }
                let dx = x - xm, dy = Double(labels[i]) - ym
                sxy += dx * dy; sxx += dx * dx; syy += dy * dy
            }
            guard sxx > 0 && syy > 0 else { continue }
            let r = sxy / (sxx.squareRoot() * syy.squareRoot())
            if abs(r) > 0.999 {
                let name = featureNames?.indices.contains(f) == true ? featureNames![f] : "\(f)"
                let w = String(format: "[MacBoost] [Warning] feature %@ correlates with the " +
                               "label at |r|=%.4f on a subsample — possible target leakage " +
                               "(is the label or a prediction included in X?)", name, abs(r))
                fitWarnings.append(w)
                progress?(w)
            }
        }
    }

    func validateCategoricals(_ X: [Float], rows: Int, cols: Int) throws {
        let dataBins = params.numBins - 1
        for f in params.categoricalFeatures {
            guard f >= 0 && f < cols else {
                throw MacBoostError.internalError("categorical feature index \(f) out of range")
            }
            for i in 0..<rows {
                let v = X[f * rows + i]
                if v.isNaN { continue }
                let c = v.rounded(.toNearestOrEven)
                if c < 0 || Int(c) >= dataBins {
                    throw MacBoostError.categoricalOutOfRange(feature: f, value: v,
                                                              maxCategories: dataBins)
                }
            }
        }
    }

    private func multiclassLogloss(predsPtr: UnsafePointer<Float>, labels: [Float],
                                   rows: Int, K: Int) -> Float {
        var s = 0.0
        for i in 0..<rows {
            var m = -Double.infinity
            for k in 0..<K { m = max(m, Double(predsPtr[k * rows + i])) }
            var sum = 0.0
            for k in 0..<K { sum += exp(Double(predsPtr[k * rows + i]) - m) }
            let c = Int(labels[i])
            let logp = Double(predsPtr[c * rows + i]) - m - log(sum)
            s -= logp
        }
        return Float(s / Double(rows))
    }

    private func trainMetric(predsPtr: UnsafeMutablePointer<Float>, labels: [Float], rows: Int) -> Float {
        labels.withUnsafeBufferPointer { lp in
            evalMetricRaw(predsPtr: predsPtr, labelsPtr: lp.baseAddress!, rows: rows)
        }
    }

    private func evalMetric(predsPtr: UnsafeMutablePointer<Float>, labels: [Float], rows: Int) -> Float {
        labels.withUnsafeBufferPointer { lp in
            evalMetricRaw(predsPtr: predsPtr, labelsPtr: lp.baseAddress!, rows: rows)
        }
    }

    private func evalMetricRaw(predsPtr: UnsafePointer<Float>, labelsPtr: UnsafePointer<Float>,
                               rows: Int) -> Float {
        switch effectiveMetric {
        case .auto, .rmse:
            var mse: Float = 0
            vDSP_distancesq(predsPtr, 1, labelsPtr, 1, &mse, vDSP_Length(rows))
            return (mse / Float(rows)).squareRoot()
        case .mae:
            var s = 0.0
            for i in 0..<rows { s += Double(abs(predsPtr[i] - labelsPtr[i])) }
            return Float(s / Double(rows))
        case .multiLogloss:
            return 0   // handled by multiclassLogloss (needs the K matrix)
        case .pinball:
            let a = Double(params.alpha)
            var s = 0.0
            for i in 0..<rows {
                let d = Double(labelsPtr[i] - predsPtr[i])
                s += d >= 0 ? a * d : (a - 1) * d
            }
            return Float(s / Double(rows))
        case .logloss:
            var s = 0.0
            for i in 0..<rows {
                let p = 1 / (1 + exp(-Double(predsPtr[i])))
                let y = Double(labelsPtr[i])
                s += -(y * log(max(p, 1e-12)) + (1 - y) * log(max(1 - p, 1e-12)))
            }
            return Float(s / Double(rows))
        case .auc:
            // Rank-sum AUC with average ranks over score ties.
            let idx = (0..<rows).sorted { predsPtr[$0] < predsPtr[$1] }
            var posRankSum = 0.0, nPos = 0.0
            var i = 0
            while i < rows {
                var j = i
                while j < rows && predsPtr[idx[j]] == predsPtr[idx[i]] { j += 1 }
                let avgRank = Double(i + j + 1) / 2   // 1-based average rank
                for k in i..<j where labelsPtr[idx[k]] > 0.5 {
                    posRankSum += avgRank; nPos += 1
                }
                i = j
            }
            let nNeg = Double(rows) - nPos
            guard nPos > 0 && nNeg > 0 else { return 0.5 }
            return Float((posRankSum - nPos * (nPos + 1) / 2) / (nPos * nNeg))
        case .poisson:
            var s = 0.0
            for i in 0..<rows {
                let mu = exp(Double(predsPtr[i])), y = Double(labelsPtr[i])
                s += 2 * ((y > 0 ? y * log(y / mu) : 0) - (y - mu))
            }
            return Float(s / Double(rows))
        case .tweedie:
            let rho = Double(params.tweedieVariancePower)
            var s = 0.0
            for i in 0..<rows {
                let mu = exp(Double(predsPtr[i])), y = Double(labelsPtr[i])
                let a = y > 0 ? pow(y, 2 - rho) / ((1 - rho) * (2 - rho)) : 0
                s += 2 * (a - y * pow(mu, 1 - rho) / (1 - rho) + pow(mu, 2 - rho) / (2 - rho))
            }
            return Float(s / Double(rows))
        }
    }

    // MARK: - Introspection

    public enum ImportanceType { case split, gain }

    /// Per-feature importance: number of splits or total gain, summed over
    /// all trees (LightGBM's importance_type="split"/"gain").
    public func featureImportance(type: ImportanceType = .gain) -> [Float] {
        var importance = [Float](repeating: 0, count: numFeatures)
        for tree in trees {
            var stack = [0]
            while let n = stack.popLast() {
                guard tree.feature[n] >= 0 else { continue }
                let f = Int(tree.feature[n])
                importance[f] += type == .split
                    ? 1 : (tree.gain.indices.contains(n) ? tree.gain[n] : 0)
                stack.append(2 * n + 1)
                stack.append(2 * n + 2)
            }
        }
        return importance
    }

    // MARK: - Inference

    struct ForestBuffers {
        let feats: MTLBuffer; let thresholds: MTLBuffer; let flags: MTLBuffer
        let leaves: MTLBuffer; let masks: MTLBuffer; let nodesPerTree: Int
    }
    private var forestCache: ForestBuffers?

    func forestBuffers() -> ForestBuffers {
        if let c = forestCache { return c }
        let nodes = trees.map { $0.feature.count }.max()!
        var feats = [Int32](); feats.reserveCapacity(trees.count * nodes)
        var thresholds = [Float](); thresholds.reserveCapacity(trees.count * nodes)
        var flags = [UInt8](); flags.reserveCapacity(trees.count * nodes)
        var leaves = [Float](); leaves.reserveCapacity(trees.count * nodes)
        var masks = [UInt32]()
        let hasCats = trees.contains { !$0.catMask.isEmpty }
        if hasCats { masks.reserveCapacity(trees.count * nodes * 8) }
        for t in trees {
            let pad = nodes - t.feature.count
            feats.append(contentsOf: t.feature)
            feats.append(contentsOf: [Int32](repeating: -1, count: pad))
            thresholds.append(contentsOf: t.threshold)
            thresholds.append(contentsOf: [Float](repeating: 0, count: pad))
            flags.append(contentsOf: t.flags)
            flags.append(contentsOf: [UInt8](repeating: 0, count: pad))
            leaves.append(contentsOf: t.leaf)
            leaves.append(contentsOf: [Float](repeating: 0, count: pad))
            if hasCats {
                masks.append(contentsOf: t.catMask.isEmpty
                    ? [UInt32](repeating: 0, count: t.feature.count * 8) : t.catMask)
                masks.append(contentsOf: [UInt32](repeating: 0, count: pad * 8))
            }
        }
        if masks.isEmpty { masks = [UInt32](repeating: 0, count: 8) }
        let c = ForestBuffers(feats: engine.makeBuffer(feats),
                              thresholds: engine.makeBuffer(thresholds),
                              flags: engine.makeBuffer(flags),
                              leaves: engine.makeBuffer(leaves),
                              masks: engine.makeBuffer(masks),
                              nodesPerTree: nodes)
        forestCache = c
        return c
    }

    /// Inference on a feature-major matrix. Returns raw scores (log-odds
    /// for binaryLogistic) except for log-link objectives (poisson,
    /// tweedie), which return the mean prediction exp(score) like LightGBM.
    /// NaN routes by the learned default direction; unseen/out-of-range
    /// categories route like missing. Large batches run on the GPU
    /// (bit-identical to the CPU path); small ones on the CPU.
    public func predict(featureMajor X: [Float], rows: Int, cols: Int) -> [Float] {
        var out = predictRawScores(featureMajor: X, rows: rows, cols: cols)
        if params.objective.usesLogLink {
            for i in 0..<out.count { out[i] = exp(out[i]) }
        }
        return out
    }

    /// Raw (untransformed) scores; used internally for warm starts.
    func predictRawScores(featureMajor X: [Float], rows: Int, cols: Int) -> [Float] {
        precondition(cols == numFeatures)
        return rows >= 4096 && !trees.isEmpty
            ? predictGPU(featureMajor: X, rows: rows, cols: cols)
            : predictCPU(featureMajor: X, rows: rows, cols: cols)
    }

    private func predictGPU(featureMajor X: [Float], rows: Int, cols: Int) -> [Float] {
        let K = trainedNumClasses
        let forest = forestBuffers()
        let xBuf = engine.makeBuffer(X)
        let outBuf = engine.makeBuffer(length: rows * K * 4)
        let baseBuf = engine.makeBuffer(baseScores)
        let cb = engine.queue.makeCommandBuffer()!
        engine.dispatch(cb, "predict_forest",
                        buffers: [xBuf, forest.feats, forest.thresholds, forest.flags,
                                  forest.leaves, forest.masks, baseBuf, outBuf],
                        params: ForestParams(numSamples: UInt32(rows),
                                             numFeatures: UInt32(cols),
                                             numTrees: UInt32(trees.count),
                                             nodesPerTree: UInt32(forest.nodesPerTree),
                                             dataBins: UInt32(trainedNumBins - 1),
                                             numClasses: UInt32(K),
                                             transform: 0),
                        grid: MTLSize(width: rows, height: 1, depth: 1),
                        threadgroup: MTLSize(width: tgSize, height: 1, depth: 1))
        cb.commit()
        cb.waitUntilCompleted()
        return [Float](UnsafeBufferPointer(
            start: outBuf.contents().bindMemory(to: Float.self, capacity: rows * K),
            count: rows * K))
    }

    func predictCPU(featureMajor X: [Float], rows: Int, cols: Int) -> [Float] {
        let K = trainedNumClasses
        if K > 1 { return predictCPUMulticlass(featureMajor: X, rows: rows, cols: cols) }
        var out = [Float](repeating: baseScore, count: rows)
        let localTrees = trees
        let dataBins = trainedNumBins - 1
        let chunk = 16_384
        let nChunks = (rows + chunk - 1) / chunk
        X.withUnsafeBufferPointer { xp in
            out.withUnsafeMutableBufferPointer { op in
                DispatchQueue.concurrentPerform(iterations: nChunks) { c in
                    for i in (c * chunk)..<min(rows, (c + 1) * chunk) {
                        for tr in localTrees {
                            var n = 0
                            while tr.feature[n] >= 0 {
                                let f = Int(tr.feature[n])
                                let v = xp[f * rows + i]
                                let left: Bool
                                if UInt32(tr.flags[n]) & flagCategorical != 0 {
                                    var cat = dataBins   // missing bin
                                    if v.isFinite {
                                        let r = v.rounded(.toNearestOrEven)
                                        if r >= 0 && Int(r) < dataBins { cat = Int(r) }
                                    }
                                    left = (tr.catMask[n * 8 + cat / 32] >> (cat % 32)) & 1 != 0
                                } else if v.isNaN {
                                    left = UInt32(tr.flags[n]) & flagDefaultLeft != 0
                                } else {
                                    left = v <= tr.threshold[n]
                                }
                                n = 2 * n + 1 + (left ? 0 : 1)
                            }
                            op[i] += tr.leaf[n]
                        }
                    }
                }
            }
        }
        return out
    }

    private func predictCPUMulticlass(featureMajor X: [Float], rows: Int,
                                      cols: Int) -> [Float] {
        let K = trainedNumClasses
        var out = [Float](repeating: 0, count: rows * K)
        let localTrees = trees
        let dataBins = trainedNumBins - 1
        X.withUnsafeBufferPointer { xp in
            out.withUnsafeMutableBufferPointer { op in
                DispatchQueue.concurrentPerform(iterations: max(1, rows / 4096 + 1)) { c in
                    let chunk = 4096
                    for i in (c * chunk)..<min(rows, (c + 1) * chunk) {
                        for k in 0..<K { op[i * K + k] = baseScores[k] }
                        for (t, tr) in localTrees.enumerated() {
                            var n = 0
                            while tr.feature[n] >= 0 {
                                let f = Int(tr.feature[n])
                                let v = xp[f * rows + i]
                                let left: Bool
                                if UInt32(tr.flags[n]) & flagCategorical != 0 {
                                    var cat = dataBins
                                    if v.isFinite {
                                        let r = v.rounded(.toNearestOrEven)
                                        if r >= 0 && Int(r) < dataBins { cat = Int(r) }
                                    }
                                    left = (tr.catMask[n * 8 + cat / 32] >> (cat % 32)) & 1 != 0
                                } else if v.isNaN {
                                    left = UInt32(tr.flags[n]) & flagDefaultLeft != 0
                                } else {
                                    left = v <= tr.threshold[n]
                                }
                                n = 2 * n + 1 + (left ? 0 : 1)
                            }
                            op[i * K + t % K] += tr.leaf[n]
                        }
                    }
                }
            }
        }
        return out
    }

    // MARK: - Serialization

    private struct ModelFile: Codable {
        var version: Int
        var objective: UInt32
        var numBins: Int
        var baseScore: Float
        var numFeatures: Int
        var categorical: [Int]
        var featureNames: [String]?
        var trees: [Tree]
        var numClasses: Int?
        var baseScores: [Float]?
    }

    public func save(to url: URL) throws {
        let m = ModelFile(version: 1, objective: params.objective.rawValue,
                          numBins: trainedNumBins, baseScore: baseScore,
                          numFeatures: numFeatures,
                          categorical: trainedCategorical.sorted(),
                          featureNames: featureNames,
                          trees: trees,
                          numClasses: trainedNumClasses,
                          baseScores: baseScores)
        try JSONEncoder().encode(m).write(to: url)
    }

    public static func load(from url: URL) throws -> MacBooster {
        let m = try JSONDecoder().decode(ModelFile.self, from: Data(contentsOf: url))
        guard m.version == 1, let obj = Objective(rawValue: m.objective) else {
            throw MacBoostError.internalError("unsupported model file")
        }
        var p = BoosterParams()
        p.numBins = m.numBins
        p.objective = obj
        p.numClasses = max(m.numClasses ?? 1, obj == .multiclass ? 3 : 1)
        p.categoricalFeatures = Set(m.categorical)
        let b = try MacBooster(params: p)
        b.trees = m.trees
        b.baseScore = m.baseScore
        b.numFeatures = m.numFeatures
        b.featureNames = m.featureNames
        b.trainedNumClasses = m.numClasses ?? 1
        b.baseScores = m.baseScores ?? [m.baseScore]
        return b
    }
}

enum Clock {
    static func now() -> UInt64 { DispatchTime.now().uptimeNanoseconds }
    static func since(_ t: UInt64) -> Double {
        Double(DispatchTime.now().uptimeNanoseconds - t) / 1e9
    }
}
