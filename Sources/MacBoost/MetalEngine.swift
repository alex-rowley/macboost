import Foundation
import Metal

public enum MacBoostError: Error, CustomStringConvertible {
    case noGPU
    case shaderCompile(String)
    case internalError(String)
    case invalidInput(String)
    case categoricalOutOfRange(feature: Int, value: Float, maxCategories: Int)

    public var description: String {
        switch self {
        case .noGPU: return "No Metal GPU available"
        case .shaderCompile(let m): return "Shader compilation failed: \(m)"
        case .internalError(let m): return "Internal error: \(m)"
        case .invalidInput(let m): return "Invalid input: \(m)"
        case .categoricalOutOfRange(let f, let v, let maxC):
            return "Categorical feature \(f) has value \(v); expected integer ids in 0..<\(maxC) or NaN"
        }
    }
}

/// Owns the Metal device, command queue and compiled compute pipelines.
/// All buffers use .storageModeShared: on Apple silicon the CPU and GPU see
/// the same physical memory, so host-side orchestration reads GPU results
/// with zero copies.
final class MetalEngine {
    let device: MTLDevice
    let queue: MTLCommandQueue
    private var pipelines: [String: MTLComputePipelineState] = [:]

    static let kernelNames = [
        "bin_data", "compute_gradients", "compute_gradients_multiclass", "quantize_gradients", "iota",
        "build_histograms", "subtract_histograms", "find_splits",
        "init_tree", "decide_splits", "final_leaves",
        "route_samples", "predict_tree_binned", "predict_forest", "gpu_treeshap", "copy_cover", "assign_leaves", "debug_bounds",
        "goss_grad_hist", "goss_threshold", "goss_select", "goss_finalize",
        "shadow_bins", "zero_buffer", "zero_built",
        "partition_count", "partition_scan", "partition_scatter", "subtract_slot",
        "copy_region", "zero_slot", "leaf_reset", "leaf_init", "leaf_pick_apply", "leaf_step",
    ]

    init() throws {
        guard let dev = MTLCreateSystemDefaultDevice() else { throw MacBoostError.noGPU }
        device = dev
        guard let q = dev.makeCommandQueue() else {
            throw MacBoostError.internalError("makeCommandQueue failed")
        }
        queue = q

        let opts = MTLCompileOptions()
        opts.mathMode = .fast
        let lib: MTLLibrary
        do {
            lib = try device.makeLibrary(source: mslSource, options: opts)
        } catch {
            throw MacBoostError.shaderCompile("\(error)")
        }
        for name in Self.kernelNames {
            guard let fn = lib.makeFunction(name: name) else {
                throw MacBoostError.shaderCompile("missing kernel \(name)")
            }
            pipelines[name] = try device.makeComputePipelineState(function: fn)
        }
    }

    func pipeline(_ name: String) -> MTLComputePipelineState { pipelines[name]! }

    func makeBuffer(length: Int) -> MTLBuffer {
        device.makeBuffer(length: max(length, 16), options: .storageModeShared)!
    }

    func makeBuffer(bytes: UnsafeRawPointer, length: Int) -> MTLBuffer {
        device.makeBuffer(bytes: bytes, length: max(length, 16),
                          options: .storageModeShared)!
    }

    func makeBuffer<T>(_ array: [T]) -> MTLBuffer {
        array.withUnsafeBytes { raw in
            device.makeBuffer(bytes: raw.baseAddress!, length: raw.count,
                              options: .storageModeShared)!
        }
    }

    /// Encode one compute dispatch. `params` is passed via setBytes at the
    /// buffer index right after the last buffer.
    func dispatch<P>(_ cb: MTLCommandBuffer, _ name: String,
                     buffers: [MTLBuffer], params: P,
                     grid: MTLSize, threadgroup: MTLSize,
                     threadgroupGrid: Bool = false, offsets: [Int]? = nil) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline(name))
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: offsets?[i] ?? 0, index: i) }
        var p = params
        enc.setBytes(&p, length: MemoryLayout<P>.stride, index: buffers.count)
        if threadgroupGrid {
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        } else {
            enc.dispatchThreads(grid, threadsPerThreadgroup: threadgroup)
        }
        enc.endEncoding()
    }

    /// Compute dispatch whose threadgroup count the GPU wrote earlier in the
    /// same command buffer (MTLDispatchThreadgroupsIndirectArguments).
    func dispatchIndirect<P>(_ cb: MTLCommandBuffer, _ name: String,
                             buffers: [MTLBuffer], params: P,
                             indirect: MTLBuffer, threadgroup: MTLSize,
                             offsets: [Int]? = nil) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline(name))
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: offsets?[i] ?? 0, index: i) }
        var p = params
        enc.setBytes(&p, length: MemoryLayout<P>.stride, index: buffers.count)
        enc.dispatchThreadgroups(indirectBuffer: indirect, indirectBufferOffset: 0,
                                 threadsPerThreadgroup: threadgroup)
        enc.endEncoding()
    }

    /// Begin a shared compute encoder: consecutive dispatches reuse one
    /// encoder with a memory barrier between them instead of paying an
    /// encoder open/close per dispatch (which dominates small-data
    /// training). Indirect dispatch arguments must be written in an
    /// EARLIER encoder than the one consuming them — group accordingly.
    func beginCompute(_ cb: MTLCommandBuffer) -> ComputeEncoding {
        ComputeEncoding(engine: self, enc: cb.makeComputeCommandEncoder()!)
    }

    func fillZero(_ cb: MTLCommandBuffer, _ buffer: MTLBuffer, length: Int) {
        let blit = cb.makeBlitCommandEncoder()!
        blit.fill(buffer: buffer, range: 0..<length, value: 0)
        blit.endEncoding()
    }

    func copyBuffer(_ cb: MTLCommandBuffer, from src: MTLBuffer, srcOffset: Int,
                    to dst: MTLBuffer, dstOffset: Int, length: Int) {
        let blit = cb.makeBlitCommandEncoder()!
        blit.copy(from: src, sourceOffset: srcOffset, to: dst,
                  destinationOffset: dstOffset, size: length)
        blit.endEncoding()
    }
}

/// One live compute encoder shared across a dependent chain of dispatches.
/// A buffer-scope memory barrier is inserted before every dispatch after
/// the first, so sequential semantics match the one-encoder-per-dispatch
/// encoding this replaces.
final class ComputeEncoding {
    private let engine: MetalEngine
    let enc: MTLComputeCommandEncoder
    private var first = true

    init(engine: MetalEngine, enc: MTLComputeCommandEncoder) {
        self.engine = engine
        self.enc = enc
    }

    private func prepare(_ name: String, _ buffers: [MTLBuffer]) {
        if !first { enc.memoryBarrier(scope: .buffers) }
        first = false
        enc.setComputePipelineState(engine.pipeline(name))
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    }

    func dispatch<P>(_ name: String, buffers: [MTLBuffer], params: P,
                     grid: MTLSize, threadgroup: MTLSize,
                     threadgroupGrid: Bool = false, offsets: [Int]? = nil) {
        prepare(name, buffers)
        if let offsets {
            for (i, o) in offsets.enumerated() where o != 0 {
                enc.setBufferOffset(o, index: i)
            }
        }
        var p = params
        enc.setBytes(&p, length: MemoryLayout<P>.stride, index: buffers.count)
        if threadgroupGrid {
            enc.dispatchThreadgroups(grid, threadsPerThreadgroup: threadgroup)
        } else {
            enc.dispatchThreads(grid, threadsPerThreadgroup: threadgroup)
        }
    }

    func dispatchIndirect<P>(_ name: String, buffers: [MTLBuffer], params: P,
                             indirect: MTLBuffer, threadgroup: MTLSize) {
        prepare(name, buffers)
        var p = params
        enc.setBytes(&p, length: MemoryLayout<P>.stride, index: buffers.count)
        enc.dispatchThreadgroups(indirectBuffer: indirect, indirectBufferOffset: 0,
                                 threadsPerThreadgroup: threadgroup)
    }

    /// Zero `length` bytes (must be a multiple of 4) via a compute kernel,
    /// so zeroing joins the shared encoder instead of forcing a blit break.
    func zero(_ buffer: MTLBuffer, length: Int) {
        dispatch("zero_buffer", buffers: [buffer], params: UInt32(length / 4),
                 grid: MTLSize(width: length / 4, height: 1, depth: 1),
                 threadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }

    func end() { enc.endEncoding() }
}
