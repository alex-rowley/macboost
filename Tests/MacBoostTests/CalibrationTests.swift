import XCTest
@testable import MacBoost

/// fold(scale:offset:) is an exact affine map on the margin, composes,
/// survives save/load, and rejects degenerate scales.
final class CalibrationTests: XCTestCase {

    func testFoldIsExactAffineMapAndComposes() throws {
        let rows = 6_000
        let (X, yRaw) = DataGen.friedman(rows: rows, cols: 6, seed: 301)
        let y = yRaw.map { $0 > 14 ? Float(1) : 0 }
        var p = BoosterParams(); p.numTrees = 40; p.maxDepth = 4; p.objective = .binaryLogistic
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: 6, labels: y)
        let before = b.predict(featureMajor: X, rows: rows, cols: 6)
        XCTAssertNil(b.calibration)

        try b.fold(scale: 0.8, offset: [-1.5])
        let after = b.predict(featureMajor: X, rows: rows, cols: 6)
        for i in stride(from: 0, to: rows, by: 97) {
            XCTAssertEqual(after[i], 0.8 * before[i] - 1.5, accuracy: 1e-4)
        }
        try b.fold(scale: 2, offset: [0.25])      // composes: 1.6·s − 2.75
        let twice = b.predict(featureMajor: X, rows: rows, cols: 6)
        for i in stride(from: 0, to: rows, by: 97) {
            XCTAssertEqual(twice[i], 1.6 * before[i] - 2.75, accuracy: 1e-4)
        }
        let c = try XCTUnwrap(b.calibration)
        XCTAssertEqual(c.folds, 2)
        XCTAssertEqual(c.scale, 1.6, accuracy: 1e-6)
        XCTAssertEqual(c.offset, [-2.75])

        let url = FileManager.default.temporaryDirectory.appendingPathComponent("fold-\(UUID()).json")
        try b.save(to: url)
        let loaded = try MacBooster.load(from: url)
        XCTAssertEqual(loaded.calibration, c)
        XCTAssertEqual(loaded.predict(featureMajor: X, rows: rows, cols: 6), twice)

        XCTAssertThrowsError(try b.fold(scale: 0, offset: [0]))
        XCTAssertThrowsError(try b.fold(scale: -1, offset: [0]))
        XCTAssertThrowsError(try b.fold(scale: 1, offset: [0, 1]))   // K = 1
    }

    func testMulticlassFoldTakesPerClassOffsets() throws {
        let rows = 6_000
        let (X, yRaw) = DataGen.friedman(rows: rows, cols: 6, seed: 302)
        let y = yRaw.map { $0 < 12 ? Float(0) : ($0 < 16 ? 1 : 2) }
        var p = BoosterParams(); p.numTrees = 30; p.maxDepth = 4
        p.objective = .multiclass; p.numClasses = 3
        let b = try MacBooster(params: p)
        try b.fit(featureMajor: X, rows: rows, cols: 6, labels: y)
        let before = b.predict(featureMajor: X, rows: rows, cols: 6)
        try b.fold(scale: 0.5, offset: [0.1, -0.2, 0.3])
        let after = b.predict(featureMajor: X, rows: rows, cols: 6)
        let off: [Float] = [0.1, -0.2, 0.3]
        for i in stride(from: 0, to: rows * 3, by: 71) {
            XCTAssertEqual(after[i], 0.5 * before[i] + off[i % 3], accuracy: 1e-4)
        }
        XCTAssertThrowsError(try b.fold(scale: 1, offset: [0, 0]))
    }
}
