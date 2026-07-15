import Foundation

struct CSVTable {
    var header: [String]
    var rowMajor: [Float]     // rows * cols, NaN for empty/non-numeric fields
    var rows: Int
    var cols: Int

    func columnIndex(_ name: String) -> Int? { header.firstIndex(of: name) }

    /// Feature-major matrix of the given columns (in the given order).
    func featureMajor(columns: [Int]) -> [Float] {
        var out = [Float](repeating: 0, count: rows * columns.count)
        out.withUnsafeMutableBufferPointer { op in
            rowMajor.withUnsafeBufferPointer { ip in
                DispatchQueue.concurrentPerform(iterations: columns.count) { j in
                    let c = columns[j]
                    for i in 0..<rows { op[j * rows + i] = ip[i * cols + c] }
                }
            }
        }
        return out
    }

    func column(_ c: Int) -> [Float] {
        var out = [Float](repeating: 0, count: rows)
        for i in 0..<rows { out[i] = rowMajor[i * cols + c] }
        return out
    }
}

enum CSVError: Error, CustomStringConvertible {
    case io(String)
    case malformed(String)

    var description: String {
        switch self {
        case .io(let m): return "CSV I/O error: \(m)"
        case .malformed(let m): return "Malformed CSV: \(m)"
        }
    }
}

enum CSV {
    /// Parse a headered numeric CSV or TSV (delimiter sniffed from the
    /// header line). Empty fields, "NA", "nan" etc. parse as NaN (missing).
    /// Parallel over rows; fields parsed with strtof.
    static func read(path: String) throws -> CSVTable {
        guard var data = FileManager.default.contents(atPath: path) else {
            throw CSVError.io("cannot read \(path)")
        }
        data.append(0x0A)   // guarantee trailing newline
        data.append(0)      // strtof terminator safety

        return try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> CSVTable in
            let bytes = raw.bindMemory(to: UInt8.self)
            let n = bytes.count - 1   // exclude NUL

            // Header line; sniff the delimiter (tab wins if present).
            var pos = 0
            while pos < n && bytes[pos] != 0x0A { pos += 1 }
            guard pos < n else { throw CSVError.malformed("no header line") }
            let headerLine = String(decoding: raw[0..<pos], as: UTF8.self)
            let delim: Character = headerLine.contains("\t") ? "\t" : ","
            let delimByte: UInt8 = delim == "\t" ? 0x09 : 0x2C
            let header = headerLine.split(separator: delim, omittingEmptySubsequences: false)
                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            let cols = header.count

            // Newline offsets for data rows.
            var lineStarts = [Int]()
            lineStarts.reserveCapacity(1 << 20)
            var p = pos + 1
            while p < n {
                if bytes[p] != 0x0A && bytes[p] != 0x0D { lineStarts.append(p) }
                while p < n && bytes[p] != 0x0A { p += 1 }
                p += 1
            }
            let rows = lineStarts.count
            guard rows > 0 else { throw CSVError.malformed("no data rows") }

            var out = [Float](repeating: .nan, count: rows * cols)
            var badRow = Int.max
            let base = raw.baseAddress!.assumingMemoryBound(to: CChar.self)
            out.withUnsafeMutableBufferPointer { op in
                let badLock = NSLock()
                DispatchQueue.concurrentPerform(iterations: rows) { i in
                    var q = lineStarts[i]
                    for c in 0..<cols {
                        // Parse one field starting at q.
                        let fieldStart = q
                        var isEnd = false
                        while true {
                            let ch = bytes[q]
                            if ch == delimByte { break }
                            if ch == 0x0A || ch == 0x0D { isEnd = true; break }
                            q += 1
                        }
                        if q > fieldStart {
                            var end: UnsafeMutablePointer<CChar>? = nil
                            let v = strtof(base + fieldStart, &end)
                            if let e = end, raw.baseAddress!.assumingMemoryBound(to: CChar.self)
                                .distance(to: e) > fieldStart {
                                op[i * cols + c] = v
                            }   // else: non-numeric -> stays NaN
                        }
                        if isEnd {
                            if c + 1 < cols {
                                badLock.lock(); badRow = min(badRow, i); badLock.unlock()
                            }
                            break
                        }
                        q += 1   // skip comma
                    }
                }
            }
            if badRow != Int.max {
                throw CSVError.malformed("row \(badRow + 1) has fewer fields than the header (\(cols))")
            }
            return CSVTable(header: header, rowMajor: out, rows: rows, cols: cols)
        }
    }

    /// Parse LibSVM / svmlight sparse text ("label idx:val idx:val ...").
    /// Index base (0 or 1) is auto-detected like sklearn; absent features
    /// are 0 (the LibSVM convention, matching LightGBM/XGBoost); qid:
    /// tokens and # comments are tolerated. Returns a dense table with
    /// header ["label", "f0", "f1", ...].
    static func readLibSVM(path: String) throws -> CSVTable {
        guard var data = FileManager.default.contents(atPath: path) else {
            throw CSVError.io("cannot read \(path)")
        }
        data.append(0x0A)
        data.append(0)

        return try data.withUnsafeBytes { (raw: UnsafeRawBufferPointer) -> CSVTable in
            let bytes = raw.bindMemory(to: UInt8.self)
            let n = bytes.count - 1
            let base = raw.baseAddress!.assumingMemoryBound(to: CChar.self)

            var lineStarts = [Int]()
            var p = 0
            while p < n {
                if bytes[p] != 0x0A && bytes[p] != 0x0D && bytes[p] != 0x23 { lineStarts.append(p) }
                while p < n && bytes[p] != 0x0A { p += 1 }
                p += 1
            }
            let rows = lineStarts.count
            guard rows > 0 else { throw CSVError.malformed("no data rows") }

            // Pass A: max/min feature index across the file (parallel chunks).
            let chunkCount = min(rows, 64)
            var chunkMax = [Int](repeating: -1, count: chunkCount)
            var chunkMin = [Int](repeating: .max, count: chunkCount)
            func forEachPair(line: Int, _ body: (Int, Float) -> Void) -> Float {
                var q = lineStarts[line]
                var end: UnsafeMutablePointer<CChar>? = nil
                let label = strtof(base + q, &end)
                q = raw.baseAddress!.assumingMemoryBound(to: CChar.self).distance(to: end!)
                while true {
                    while bytes[q] == 0x20 || bytes[q] == 0x09 { q += 1 }
                    if bytes[q] == 0x0A || bytes[q] == 0x0D || bytes[q] == 0x23 { break }
                    var e: UnsafeMutablePointer<CChar>? = nil
                    let idx = strtol(base + q, &e, 10)
                    let idxEnd = base.distance(to: e!)
                    if idxEnd == q || bytes[idxEnd] != 0x3A {   // not "num:" (e.g. qid:)
                        var skip = q
                        while bytes[skip] != 0x20 && bytes[skip] != 0x09
                                && bytes[skip] != 0x0A && bytes[skip] != 0x0D { skip += 1 }
                        q = skip
                        continue
                    }
                    var ve: UnsafeMutablePointer<CChar>? = nil
                    let v = strtof(base + idxEnd + 1, &ve)
                    q = base.distance(to: ve!)
                    body(idx, v)
                }
                return label
            }
            chunkMax.withUnsafeMutableBufferPointer { mx in
                chunkMin.withUnsafeMutableBufferPointer { mn in
                    DispatchQueue.concurrentPerform(iterations: chunkCount) { c in
                        var lo = c * rows / chunkCount
                        let hi = (c + 1) * rows / chunkCount
                        var localMax = -1, localMin = Int.max
                        while lo < hi {
                            _ = forEachPair(line: lo) { idx, _ in
                                localMax = max(localMax, idx); localMin = min(localMin, idx)
                            }
                            lo += 1
                        }
                        mx[c] = localMax; mn[c] = localMin

                    }
                }
            }
            let maxIdx = chunkMax.max() ?? -1
            let minIdx = chunkMin.min() ?? .max
            guard maxIdx >= 0 else { throw CSVError.malformed("no idx:value pairs found") }
            let indexBase = minIdx == 0 ? 0 : 1          // sklearn-style auto detection
            let features = maxIdx - indexBase + 1
            let cols = features + 1                       // + label column 0

            // Pass B: fill the dense matrix (absent = 0 per LibSVM convention).
            var out = [Float](repeating: 0, count: rows * cols)
            out.withUnsafeMutableBufferPointer { op in
                DispatchQueue.concurrentPerform(iterations: rows) { i in
                    let label = forEachPair(line: i) { idx, v in
                        op[i * cols + 1 + (idx - indexBase)] = v
                    }
                    op[i * cols] = label
                }
            }
            let header = ["label"] + (0..<features).map { "f\($0)" }
            return CSVTable(header: header, rowMajor: out, rows: rows, cols: cols)
        }
    }

    static func writePredictions(_ preds: [Float], to path: String, header: String) throws {
        var s = header + "\n"
        s.reserveCapacity(preds.count * 12)
        for v in preds { s += "\(v)\n" }
        try s.write(toFile: path, atomically: true, encoding: .utf8)
    }
}
