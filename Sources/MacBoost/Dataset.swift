import Foundation

/// A preprocessed training dataset: the quantile-binned (tiled) feature
/// matrix, bin edges, labels and metadata — LightGBM's `save_binary`
/// pattern. Parsing and binning are paid once; `MacBooster.fit(dataset:)`
/// then starts training immediately, which makes hyperparameter sweeps and
/// repeated runs on large files near-instant.
///
/// The file locks the bin count and categorical layout it was built with;
/// fit(dataset:) uses those, not the booster's params.
public struct BinnedDataset {
    public let rows: Int
    public let cols: Int
    public let numBins: Int
    public let categorical: Set<Int>
    public let featureNames: [String]?
    public let labels: [Float]
    let bins: [UInt8]          // tiled layout (see Shaders.swift header)
    let edges: [Float]         // cols * (numBins - 2)

    static let magic: UInt32 = 0x4D424453    // "MBDS"
    static let version: UInt32 = 1

    // MARK: binary serialization (little-endian, Apple-silicon native)

    public func save(to url: URL) throws {
        var d = Data()
        func put<T: FixedWidthInteger>(_ v: T) { withUnsafeBytes(of: v.littleEndian) { d.append(contentsOf: $0) } }
        put(Self.magic); put(Self.version); put(UInt32(8))   // tileF
        put(UInt64(rows)); put(UInt32(cols)); put(UInt32(numBins))
        put(UInt32(categorical.count))
        for c in categorical.sorted() { put(UInt32(c)) }
        let namesData = try featureNames.map { try JSONEncoder().encode($0) } ?? Data()
        put(UInt32(namesData.count)); d.append(namesData)
        put(UInt64(edges.count));  edges.withUnsafeBytes { d.append(contentsOf: $0) }
        put(UInt64(labels.count)); labels.withUnsafeBytes { d.append(contentsOf: $0) }
        put(UInt64(bins.count));   d.append(contentsOf: bins)
        try d.write(to: url)
    }

    public static func load(from url: URL) throws -> BinnedDataset {
        let d = try Data(contentsOf: url)
        var pos = 0
        func take(_ n: Int) throws -> Data {
            guard pos + n <= d.count else {
                throw MacBoostError.invalidInput("truncated .mbds file")
            }
            defer { pos += n }
            return d.subdata(in: pos..<(pos + n))
        }
        func u32() throws -> UInt32 { try take(4).withUnsafeBytes { $0.loadUnaligned(as: UInt32.self) } }
        func u64() throws -> UInt64 { try take(8).withUnsafeBytes { $0.loadUnaligned(as: UInt64.self) } }

        guard try u32() == magic else {
            throw MacBoostError.invalidInput("not a MacBoost dataset (.mbds) file")
        }
        guard try u32() == version else {
            throw MacBoostError.invalidInput("unsupported .mbds version")
        }
        guard try u32() == 8 else {
            throw MacBoostError.invalidInput(".mbds tile width mismatch")
        }
        let rows = Int(try u64()), cols = Int(try u32()), numBins = Int(try u32())
        let catCount = Int(try u32())
        var categorical = Set<Int>()
        for _ in 0..<catCount { categorical.insert(Int(try u32())) }
        let namesLen = Int(try u32())
        let featureNames: [String]? = namesLen > 0
            ? try JSONDecoder().decode([String].self, from: try take(namesLen)) : nil

        let edgeCount = Int(try u64())
        let edges: [Float] = try take(edgeCount * 4).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let labelCount = Int(try u64())
        let labels: [Float] = try take(labelCount * 4).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
        let binCount = Int(try u64())
        let bins = [UInt8](try take(binCount))

        let numTiles = (cols + 7) / 8
        guard labelCount == rows, binCount == rows * numTiles * 8,
              edgeCount == cols * (numBins - 2) else {
            throw MacBoostError.invalidInput(".mbds sizes are inconsistent")
        }
        return BinnedDataset(rows: rows, cols: cols, numBins: numBins,
                             categorical: categorical, featureNames: featureNames,
                             labels: labels, bins: bins, edges: edges)
    }
}
