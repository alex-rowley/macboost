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
                     threadgroupGrid: Bool = false) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline(name))
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
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
                             indirect: MTLBuffer, threadgroup: MTLSize) {
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(pipeline(name))
        for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
        var p = params
        enc.setBytes(&p, length: MemoryLayout<P>.stride, index: buffers.count)
        enc.dispatchThreadgroups(indirectBuffer: indirect, indirectBufferOffset: 0,
                                 threadsPerThreadgroup: threadgroup)
        enc.endEncoding()
    }

    func fillZero(_ cb: MTLCommandBuffer, _ buffer: MTLBuffer, length: Int) {
        let blit = cb.makeBlitCommandEncoder()!
        blit.fill(buffer: buffer, range: 0..<length, value: 0)
        blit.endEncoding()
    }
}
