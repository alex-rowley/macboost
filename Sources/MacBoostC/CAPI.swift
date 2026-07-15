import Foundation
import MacBoost

// C ABI for language bindings (Python via ctypes). Handles are retained
// MacBooster instances; matrices are feature-major float32. All entry
// points report failures through an out-parameter error string that the
// caller frees with macboost_free_string.

private struct CConfig: Codable {
    var num_trees: Int?
    var max_depth: Int?
    var learning_rate: Float?
    var lambda: Float?
    var min_child_weight: Float?
    var min_split_gain: Float?
    var num_bins: Int?
    var objective: String?
    var categorical_features: [Int]?
    var cat_smooth: Float?
    var early_stopping_rounds: Int?
    var eval_every: Int?
    var verbose: Bool?
    var goss: Bool?
    var goss_top_rate: Float?
    var goss_other_rate: Float?
    var alpha: Float?
    var tweedie_variance_power: Float?
    var scale_pos_weight: Float?
    var subsample: Float?
    var feature_fraction: Float?
    var monotone_constraints: [Int]?
    var metric: String?
    var num_class: Int?
    var init_model: String?
}

private func setError(_ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?,
                      _ message: String) {
    err?.pointee = strdup(message)
}

private func booster(_ handle: UnsafeMutableRawPointer) -> MacBooster {
    Unmanaged<MacBooster>.fromOpaque(handle).takeUnretainedValue()
}

@_cdecl("macboost_train")
public func macboost_train(
    _ paramsJSON: UnsafePointer<CChar>?,
    _ x: UnsafePointer<Float>?, _ rows: Int64, _ cols: Int64,
    _ y: UnsafePointer<Float>?,
    _ w: UnsafePointer<Float>?,
    _ xValid: UnsafePointer<Float>?, _ validRows: Int64, _ yValid: UnsafePointer<Float>?,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutableRawPointer? {
    do {
        guard let x, let y, rows > 0, cols > 0 else {
            throw MacBoostError.internalError("null or empty training data")
        }
        var cfg = CConfig()
        if let paramsJSON {
            let data = Data(String(cString: paramsJSON).utf8)
            cfg = try JSONDecoder().decode(CConfig.self, from: data)
        }
        var p = BoosterParams()
        if let v = cfg.num_trees { p.numTrees = v }
        if let v = cfg.max_depth { p.maxDepth = v }
        if let v = cfg.learning_rate { p.learningRate = v }
        if let v = cfg.lambda { p.lambda = v }
        if let v = cfg.min_child_weight { p.minChildHess = v }
        if let v = cfg.min_split_gain { p.minSplitGain = v }
        if let v = cfg.num_bins { p.numBins = v }
        if let v = cfg.categorical_features { p.categoricalFeatures = Set(v) }
        if let v = cfg.cat_smooth { p.catSmooth = v }
        if let v = cfg.goss { p.goss = v }
        if let v = cfg.goss_top_rate { p.gossTopRate = v }
        if let v = cfg.goss_other_rate { p.gossOtherRate = v }
        if let v = cfg.alpha { p.alpha = v }
        if let v = cfg.tweedie_variance_power { p.tweedieVariancePower = v }
        if let v = cfg.scale_pos_weight { p.scalePosWeight = v }
        if let v = cfg.subsample { p.subsample = v }
        if let v = cfg.feature_fraction { p.featureFraction = v }
        if let v = cfg.monotone_constraints { p.monotoneConstraints = v }
        if let v = cfg.num_class { p.numClasses = v }
        if let v = cfg.metric, let m = EvalMetric(rawValue: v) { p.metric = m }
        switch cfg.objective ?? "regression" {
        case "regression", "l2", "mse": p.objective = .regression
        case "binary": p.objective = .binaryLogistic
        case "mae", "l1", "regression_l1": p.objective = .mae
        case "huber": p.objective = .huber
        case "quantile": p.objective = .quantile
        case "poisson": p.objective = .poisson
        case "tweedie": p.objective = .tweedie
        case "multiclass", "softmax": p.objective = .multiclass
        case let o: throw MacBoostError.internalError("unknown objective '\(o)'")
        }

        let b = try MacBooster(params: p)
        let X = Array(UnsafeBufferPointer(start: x, count: Int(rows * cols)))
        let Y = Array(UnsafeBufferPointer(start: y, count: Int(rows)))
        var evalSet: EvalSet?
        if let xValid, let yValid, validRows > 0 {
            evalSet = EvalSet(
                featureMajor: Array(UnsafeBufferPointer(start: xValid,
                                                        count: Int(validRows * cols))),
                rows: Int(validRows),
                labels: Array(UnsafeBufferPointer(start: yValid, count: Int(validRows))))
        }
        let verbose = cfg.verbose ?? false
        let weights = w.map { Array(UnsafeBufferPointer(start: $0, count: Int(rows))) }
        let initModel = try cfg.init_model.map {
            try MacBooster.load(from: URL(fileURLWithPath: $0))
        }
        try b.fit(featureMajor: X, rows: Int(rows), cols: Int(cols), labels: Y,
                  weights: weights, valid: evalSet,
                  earlyStoppingRounds: cfg.early_stopping_rounds ?? 0,
                  evalEvery: cfg.eval_every ?? 0,
                  initModel: initModel,
                  progress: verbose ? { print($0) } : nil)
        return Unmanaged.passRetained(b).toOpaque()
    } catch {
        setError(err, "\(error)")
        return nil
    }
}

@_cdecl("macboost_predict")
public func macboost_predict(
    _ handle: UnsafeMutableRawPointer?,
    _ x: UnsafePointer<Float>?, _ rows: Int64, _ cols: Int64,
    _ out: UnsafeMutablePointer<Float>?,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let handle, let x, let out else { setError(err, "null argument"); return -1 }
    let b = booster(handle)
    guard Int(cols) == b.numFeatures else {
        setError(err, "model expects \(b.numFeatures) features, got \(cols)")
        return -1
    }
    let X = Array(UnsafeBufferPointer(start: x, count: Int(rows * cols)))
    let preds = b.predict(featureMajor: X, rows: Int(rows), cols: Int(cols))
    preds.withUnsafeBufferPointer { out.update(from: $0.baseAddress!, count: $0.count) }
    return 0
}

@_cdecl("macboost_num_classes")
public func macboost_num_classes(_ handle: UnsafeMutableRawPointer?) -> Int64 {
    guard let handle else { return -1 }
    return Int64(booster(handle).trainedNumClasses)
}

/// type: 0 = gain, 1 = split. `out` must hold num_features floats.
@_cdecl("macboost_feature_importance")
public func macboost_feature_importance(
    _ handle: UnsafeMutableRawPointer?, _ type: Int32,
    _ out: UnsafeMutablePointer<Float>?
) -> Int32 {
    guard let handle, let out else { return -1 }
    let imp = booster(handle).featureImportance(type: type == 1 ? .split : .gain)
    imp.withUnsafeBufferPointer { out.update(from: $0.baseAddress!, count: $0.count) }
    return 0
}

/// SHAP contributions; `out` must hold rows * (cols + 1) * num_classes floats.
@_cdecl("macboost_predict_contrib")
public func macboost_predict_contrib(
    _ handle: UnsafeMutableRawPointer?,
    _ x: UnsafePointer<Float>?, _ rows: Int64, _ cols: Int64,
    _ out: UnsafeMutablePointer<Float>?,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let handle, let x, let out else { setError(err, "null argument"); return -1 }
    let b = booster(handle)
    guard Int(cols) == b.numFeatures else {
        setError(err, "model expects \(b.numFeatures) features, got \(cols)")
        return -1
    }
    let X = Array(UnsafeBufferPointer(start: x, count: Int(rows * cols)))
    let c = b.predictContributions(featureMajor: X, rows: Int(rows), cols: Int(cols))
    c.withUnsafeBufferPointer { out.update(from: $0.baseAddress!, count: $0.count) }
    return 0
}

@_cdecl("macboost_save")
public func macboost_save(
    _ handle: UnsafeMutableRawPointer?, _ path: UnsafePointer<CChar>?,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> Int32 {
    guard let handle, let path else { setError(err, "null argument"); return -1 }
    do {
        try booster(handle).save(to: URL(fileURLWithPath: String(cString: path)))
        return 0
    } catch {
        setError(err, "\(error)")
        return -1
    }
}

@_cdecl("macboost_load")
public func macboost_load(
    _ path: UnsafePointer<CChar>?,
    _ err: UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>?
) -> UnsafeMutableRawPointer? {
    guard let path else { setError(err, "null path"); return nil }
    do {
        let b = try MacBooster.load(from: URL(fileURLWithPath: String(cString: path)))
        return Unmanaged.passRetained(b).toOpaque()
    } catch {
        setError(err, "\(error)")
        return nil
    }
}

@_cdecl("macboost_num_trees")
public func macboost_num_trees(_ handle: UnsafeMutableRawPointer?) -> Int64 {
    guard let handle else { return -1 }
    return Int64(booster(handle).trees.count)
}

@_cdecl("macboost_num_features")
public func macboost_num_features(_ handle: UnsafeMutableRawPointer?) -> Int64 {
    guard let handle else { return -1 }
    return Int64(booster(handle).numFeatures)
}

@_cdecl("macboost_best_iteration")
public func macboost_best_iteration(_ handle: UnsafeMutableRawPointer?) -> Int64 {
    guard let handle else { return -1 }
    return Int64(booster(handle).bestIteration ?? 0)
}

/// Newline-joined data-quality warnings from the last fit, or NULL if none.
/// Caller frees with macboost_free_string.
@_cdecl("macboost_warnings")
public func macboost_warnings(_ handle: UnsafeMutableRawPointer?) -> UnsafeMutablePointer<CChar>? {
    guard let handle else { return nil }
    let w = booster(handle).fitWarnings
    return w.isEmpty ? nil : strdup(w.joined(separator: "\n"))
}

@_cdecl("macboost_free")
public func macboost_free(_ handle: UnsafeMutableRawPointer?) {
    guard let handle else { return }
    Unmanaged<MacBooster>.fromOpaque(handle).release()
}

@_cdecl("macboost_free_string")
public func macboost_free_string(_ s: UnsafeMutablePointer<CChar>?) {
    free(s)
}
