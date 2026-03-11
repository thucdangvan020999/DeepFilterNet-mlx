import DeepFilterNetMLX
import Foundation
import MLX

/// Wraps the existing MLX GPU streaming implementation for benchmarking.
public final class MLXStreamingEngine: StreamingEngine {
    public let engineType: StreamingEngineType = .mlxGPU
    public var supportsOffline: Bool { true }

    private var streamer: DeepFilterNetModel.DeepFilterNetStreamer?
    private var model: DeepFilterNetModel?

    public init() {}

    public func prepare(model: DeepFilterNetModel) throws -> Double {
        let t0 = CFAbsoluteTimeGetCurrent()
        self.model = model
        model.configurePerformance(.throughput)
        self.streamer = model.createStreamer(config: DeepFilterNetStreamingConfig(
            padEndFrames: 3,
            compensateDelay: true,
            enableStageSkipping: false,
            materializeEveryHops: 96
        ))
        return CFAbsoluteTimeGetCurrent() - t0
    }

    public func processHop(_ samples: [Float]) throws -> [Float] {
        guard let streamer else {
            throw BenchmarkError.engineNotPrepared
        }
        return try streamer.processChunk(samples)
    }

    public func flush() throws -> [Float] {
        guard let streamer else {
            throw BenchmarkError.engineNotPrepared
        }
        return try streamer.flush()
    }

    public func reset() {
        streamer?.reset()
    }

    public func enhanceOffline(_ audio: MLXArray, model: DeepFilterNetModel) throws -> MLXArray {
        model.configurePerformance(.throughput)
        return try model.enhance(audio)
    }
}

public enum BenchmarkError: Error, LocalizedError {
    case engineNotPrepared
    case coreMLModelNotFound(String)
    case unsupportedModelVersion

    public var errorDescription: String? {
        switch self {
        case .engineNotPrepared:
            return "Engine not prepared. Call prepare(model:) first."
        case .coreMLModelNotFound(let path):
            return "CoreML model not found at: \(path)"
        case .unsupportedModelVersion:
            return "This engine does not support the given model version."
        }
    }
}
