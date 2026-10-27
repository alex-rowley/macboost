import Accelerate
import Foundation

/// One affine block: outputs = matrix · (x[inputs] − mean).
/// `matrix` is outputs × inputs.count, row-major.
public struct ProjectionBlock: Codable, Equatable {
    public var inputs: [Int]
    public var mean: [Float]
    public var matrix: [Float]
    public var outputs: Int

    public init(inputs: [Int], mean: [Float], matrix: [Float], outputs: Int) {
        self.inputs = inputs; self.mean = mean; self.matrix = matrix; self.outputs = outputs
    }
}

/// Affine input map stored with the model so raw-width X goes in at fit and
/// predict. Projected columns are the blocks' outputs in order, then the
/// passthrough columns. `fill` (optional, one per raw column) replaces NaN
/// before projection; a NaN entry leaves that column's NaN alone.
public struct InputProjection: Codable, Equatable {
    public var inputWidth: Int
    public var blocks: [ProjectionBlock]
    public var passthrough: [Int]
    public var fill: [Float?]?      // nil entry = leave NaN alone

    public init(inputWidth: Int, blocks: [ProjectionBlock], passthrough: [Int] = [],
                fill: [Float?]? = nil) {
        self.inputWidth = inputWidth; self.blocks = blocks
        self.passthrough = passthrough; self.fill = fill
    }

    public var outputWidth: Int { blocks.reduce(0) { $0 + $1.outputs } + passthrough.count }

    func validate() throws {
        guard inputWidth > 0 else { throw MacBoostError.invalidInput("projection inputWidth must be > 0") }
        guard outputWidth > 0 else { throw MacBoostError.invalidInput("projection produces no columns") }
        if let fill, fill.count != inputWidth {
            throw MacBoostError.invalidInput("projection fill needs \(inputWidth) values, got \(fill.count)")
        }
        for (i, b) in blocks.enumerated() {
            guard !b.inputs.isEmpty, b.outputs > 0 else {
                throw MacBoostError.invalidInput("projection block \(i) is empty")
            }
            guard b.mean.count == b.inputs.count else {
                throw MacBoostError.invalidInput("projection block \(i): mean has \(b.mean.count) values for \(b.inputs.count) inputs")
            }
            guard b.matrix.count == b.outputs * b.inputs.count else {
                throw MacBoostError.invalidInput("projection block \(i): matrix has \(b.matrix.count) values, expected \(b.outputs)×\(b.inputs.count)")
            }
            guard b.inputs.allSatisfy({ $0 >= 0 && $0 < inputWidth }),
                  b.matrix.allSatisfy({ $0.isFinite }), b.mean.allSatisfy({ $0.isFinite }) else {
                throw MacBoostError.invalidInput("projection block \(i): input index out of range or non-finite values")
            }
        }
        guard passthrough.allSatisfy({ $0 >= 0 && $0 < inputWidth }) else {
            throw MacBoostError.invalidInput("projection passthrough index out of range")
        }
    }

    /// Project a view into a feature-major matrix of outputWidth columns.
    func apply(_ X: MatrixView) -> [Float] {
        let rows = X.rows
        var out = [Float](repeating: 0, count: rows * outputWidth)
        var col = 0
        for b in blocks {
            let k = b.inputs.count
            var centred = [Float](repeating: 0, count: rows * k)   // column-major rows×k
            for (j, f) in b.inputs.enumerated() {
                let fv = fill?[f] ?? nil ?? .nan
                let mu = b.mean[j]
                for i in 0..<rows {
                    var v = X.value(i, f)
                    if v.isNaN && !fv.isNaN { v = fv }
                    centred[j * rows + i] = v - mu
                }
            }
            // out[rows×outputs] = centred[rows×k] · matrixᵀ[k×outputs]; matrix is
            // row-major outputs×k, which is column-major k×outputs: no transpose.
            centred.withUnsafeBufferPointer { a in
                b.matrix.withUnsafeBufferPointer { w in
                    out.withUnsafeMutableBufferPointer { o in
                        cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                                    Int32(rows), Int32(b.outputs), Int32(k),
                                    1, a.baseAddress!, Int32(rows),
                                    w.baseAddress!, Int32(k),
                                    0, o.baseAddress! + col * rows, Int32(rows))
                    }
                }
            }
            col += b.outputs
        }
        for f in passthrough {
            let fv = fill?[f] ?? nil ?? .nan
            for i in 0..<rows {
                var v = X.value(i, f)
                if v.isNaN && !fv.isNaN { v = fv }
                out[col * rows + i] = v
            }
            col += 1
        }
        return out
    }

    func apply(featureMajor X: [Float], rows: Int) -> [Float] {
        X.withUnsafeBufferPointer {
            apply(MatrixView(base: $0.baseAddress!, rows: rows, cols: inputWidth, rowMajor: false))
        }
    }
}
