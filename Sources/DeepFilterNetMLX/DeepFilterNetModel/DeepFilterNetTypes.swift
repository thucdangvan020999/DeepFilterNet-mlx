import Foundation
import HuggingFace
import MLX
import MLXNN

public enum DeepFilterNetError: Error, LocalizedError, CustomStringConvertible {
    case invalidRepoID(String)
    case modelPathNotFound(String)
    case missingConfig(URL)
    case missingWeights(URL)
    case missingWeightKey(String)
    case invalidAudioShape([Int])
    case streamingNotSupportedForModelVersion(String)

    public var errorDescription: String? { description }

    public var description: String {
        switch self {
        case .invalidRepoID(let value):
            return "Invalid Hugging Face model repo ID: \(value)"
        case .modelPathNotFound(let path):
            return "Model path not found: \(path)"
        case .missingConfig(let directory):
            return "Missing config.json in model directory: \(directory.path)"
        case .missingWeights(let directory):
            return "Missing .safetensors weights in model directory: \(directory.path)"
        case .missingWeightKey(let key):
            return "Missing DeepFilterNet weight key: \(key)"
        case .invalidAudioShape(let shape):
            return "Expected mono 1D audio array, got shape: \(shape)"
        case .streamingNotSupportedForModelVersion(let version):
            return "Streaming is not supported for model version \(version). Use offline enhancement instead."
        }
    }
}

public enum DeepFilterNetPrecision: String, CaseIterable, Sendable {
    case fp32
    case fp16

    var dtype: DType {
        switch self {
        case .fp32:
            return .float32
        case .fp16:
            return .float16
        }
    }
}

public struct DeepFilterNetStreamingConfig: Sendable {
    public var padEndFrames: Int
    public var compensateDelay: Bool
    public var enableStageSkipping: Bool
    public var minDbThresh: Float
    public var maxDbErbThresh: Float
    public var maxDbDfThresh: Float
    public var enableProfiling: Bool
    public var profilingForceEvalPerStage: Bool
    public var materializeEveryHops: Int

    public init(
        padEndFrames: Int = 3,
        compensateDelay: Bool = true,
        enableStageSkipping: Bool = false,
        minDbThresh: Float = -10.0,
        maxDbErbThresh: Float = 30.0,
        maxDbDfThresh: Float = 20.0,
        enableProfiling: Bool = false,
        profilingForceEvalPerStage: Bool = false,
        materializeEveryHops: Int = 96
    ) {
        self.padEndFrames = padEndFrames
        self.compensateDelay = compensateDelay
        self.enableStageSkipping = enableStageSkipping
        self.minDbThresh = minDbThresh
        self.maxDbErbThresh = maxDbErbThresh
        self.maxDbDfThresh = maxDbDfThresh
        self.enableProfiling = enableProfiling
        self.profilingForceEvalPerStage = profilingForceEvalPerStage
        self.materializeEveryHops = materializeEveryHops
    }
}

public struct DeepFilterNetStreamingChunk: @unchecked Sendable {
    public let audio: MLXArray
    public let chunkIndex: Int
    public let isLastChunk: Bool
}
