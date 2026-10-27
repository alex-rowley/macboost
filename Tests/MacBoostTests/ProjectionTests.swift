import XCTest
@testable import MacBoost

/// A stored affine input map is applied identically at fit and predict,
/// matches pre-projected training, round-trips, and validates its shape.
final class ProjectionTests: XCTestCase {

    /// Feature-major synthetic: 6 raw columns; the model sees a 2-D rotation
    /// of columns 0–2 plus column 5 passed through.
    func makeData(rows: Int, seed: UInt64) -> (X: [Float], y: [Float]) {
        var rng = SplitMix64(seed: seed)
        var X = [Float](repeating: 0, count: rows * 6)
        var y = [Float](repeating: 0, count: rows)
        for i in 0..<rows {
            let a = Float(rng.uniform()), b = Float(rng.uniform()), c = Float(rng.uniform())
            X[0 * rows + i] = a + b; X[1 * rows + i] = a - b; X[2 * rows + i] = c
            X[3 * rows + i] = Float(rng.uniform()); X[4 * rows + i] = Float(rng.uniform())
            X[5 * rows + i] = Float(rng.uniform())
            y[i] = 3 * a + 2 * c + X[5 * rows + i] + 0.1 * Float(rng.uniform())
        }
        return (X, y)
    }

    func projection() -> InputProjection {
        // Recovers a = (x0 + x1)/2 and c = x2 from columns 0..2.
        InputProjection(inputWidth: 6,
                        blocks: [ProjectionBlock(inputs: [0, 1, 2], mean: [1, 0, 0.5],
                                                 matrix: [0.5, 0.5, 0, 0, 0, 1], outputs: 2)],
                        passthrough: [5])
    }

    func testProjectedFitMatchesPreProjectedAndRoundTrips() throws {
        let rows = 8_000
        let (X, y) = makeData(rows: rows, seed: 401)
        var p = BoosterParams(); p.numTrees = 40; p.maxDepth = 4
        let proj = projection()

        let b = try MacBooster(params: p)
        try b.setProjection(proj)
        try b.fit(featureMajor: X, rows: rows, cols: 6, labels: y)
        XCTAssertEqual(b.numFeatures, 3)
        XCTAssertEqual(b.inputWidth, 6)

        let Z = X.withUnsafeBufferPointer {
            proj.apply(MatrixView(base: $0.baseAddress!, rows: rows, cols: 6, rowMajor: false))
        }
        let ref = try MacBooster(params: p)
        try ref.fit(featureMajor: Z, rows: rows, cols: 3, labels: y)
        let predRaw = b.predict(featureMajor: X, rows: rows, cols: 6)
        let refPred = ref.predict(featureMajor: Z, rows: rows, cols: 3)
        for i in stride(from: 0, to: rows, by: 53) {
            XCTAssertEqual(predRaw[i], refPred[i], accuracy: 1e-4)
        }
        let shap = b.predictContributions(featureMajor: X, rows: 64, cols: 6)
        XCTAssertEqual(shap.count, 64 * 4, "contributions are in projected space (+ bias)")

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("proj-\(UUID()).json")
        try b.save(to: url)
        let loaded = try MacBooster.load(from: url)
        XCTAssertEqual(loaded.projection, proj)
        XCTAssertEqual(loaded.predict(featureMajor: X, rows: rows, cols: 6), predRaw)
    }

    /// A full-rank (square) projection cannot be told apart from raw input
    /// by width, so the rule is: with a projection attached, input is raw.
    func testSquareProjectionIsAlwaysApplied() throws {
        let rows = 6_000
        let (X, y) = makeData(rows: rows, seed: 404)
        // 6×6 rotation mixing columns 0,1 (a+b, a−b -> a, b) and permuting the rest.
        let proj = InputProjection(inputWidth: 6, blocks: [
            ProjectionBlock(inputs: [0, 1, 2, 3, 4, 5], mean: [0, 0, 0, 0, 0, 0],
                            matrix: [0.5, 0.5, 0, 0, 0, 0,
                                     0.5, -0.5, 0, 0, 0, 0,
                                     0, 0, 0, 0, 0, 1,
                                     0, 0, 1, 0, 0, 0,
                                     0, 0, 0, 1, 0, 0,
                                     0, 0, 0, 0, 1, 0], outputs: 6)])
        var p = BoosterParams(); p.numTrees = 30; p.maxDepth = 4
        let b = try MacBooster(params: p)
        try b.setProjection(proj)
        try b.fit(featureMajor: X, rows: rows, cols: 6, labels: y)
        let Z = X.withUnsafeBufferPointer {
            proj.apply(MatrixView(base: $0.baseAddress!, rows: rows, cols: 6, rowMajor: false))
        }
        let ref = try MacBooster(params: p)
        try ref.fit(featureMajor: Z, rows: rows, cols: 6, labels: y)
        let pred = b.predict(featureMajor: X, rows: rows, cols: 6)
        let refPred = ref.predict(featureMajor: Z, rows: rows, cols: 6)
        var maxGap: Float = 0
        for i in 0..<rows { maxGap = max(maxGap, abs(pred[i] - refPred[i])) }
        XCTAssertLessThan(maxGap, 0.05, "square projection must be applied at predict")
        var X32 = [Float](repeating: 0, count: 32 * 6)      // feature-major slice
        for j in 0..<6 { for i in 0..<32 { X32[j * 32 + i] = X[j * rows + i] } }
        let shap = b.predictContributions(featureMajor: X32, rows: 32, cols: 6)
        var sums = [Float](repeating: 0, count: 32)
        for i in 0..<32 { for j in 0..<7 { sums[i] += shap[i * 7 + j] } }
        for i in 0..<32 { XCTAssertEqual(sums[i], pred[i], accuracy: 1e-3) }
    }

    func testProjectionValidation() throws {
        var p = BoosterParams(); p.numTrees = 5
        let b = try MacBooster(params: p)
        XCTAssertThrowsError(try b.setProjection(
            InputProjection(inputWidth: 6, blocks: [ProjectionBlock(inputs: [0, 9], mean: [0, 0],
                                                                     matrix: [1, 0], outputs: 1)])))
        XCTAssertThrowsError(try b.setProjection(
            InputProjection(inputWidth: 6, blocks: [ProjectionBlock(inputs: [0, 1], mean: [0],
                                                                     matrix: [1, 0], outputs: 1)])))
        XCTAssertThrowsError(try b.setProjection(InputProjection(inputWidth: 6, blocks: [])))
        // Width mismatch at fit.
        try b.setProjection(projection())
        let (X, y) = makeData(rows: 500, seed: 402)
        XCTAssertThrowsError(try b.fit(featureMajor: Array(X[0..<(500 * 5)]), rows: 500, cols: 5, labels: y))
        // Attaching to a trained model requires the trees' width.
        let (X2, y2) = makeData(rows: 2_000, seed: 403)
        let t = try MacBooster(params: p)
        try t.fit(featureMajor: X2, rows: 2_000, cols: 6, labels: y2)
        XCTAssertThrowsError(try t.setProjection(projection()))   // produces 3, model has 6
    }

    func testFillReplacesNaNBeforeProjection() throws {
        let rows = 4
        var X = [Float](repeating: 1, count: rows * 2)
        X[0] = .nan                                       // column 0, row 0
        var proj = InputProjection(inputWidth: 2,
                                   blocks: [ProjectionBlock(inputs: [0, 1], mean: [0, 0],
                                                            matrix: [1, 1], outputs: 1)])
        let withoutFill = X.withUnsafeBufferPointer {
            proj.apply(MatrixView(base: $0.baseAddress!, rows: rows, cols: 2, rowMajor: false))
        }
        XCTAssertTrue(withoutFill[0].isNaN)
        proj.fill = [5, nil]
        let withFill = X.withUnsafeBufferPointer {
            proj.apply(MatrixView(base: $0.baseAddress!, rows: rows, cols: 2, rowMajor: false))
        }
        XCTAssertEqual(withFill[0], 6)
        XCTAssertEqual(withFill[1], 2)
    }
}
