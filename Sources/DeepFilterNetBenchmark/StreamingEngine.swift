import DeepFilterNetMLX
import Foundation
import MLX

// MARK: - Streaming Engine Protocol

/// Common interface for all streaming engine implementations.
/// Each engine processes audio hop-by-hop and returns enhanced output.
public protocol StreamingEngine: AnyObject {
    /// Which engine type this is
    var engineType: StreamingEngineType { get }

    /// Initialize the engine for streaming (extract weights, build state, etc.)
    /// Returns the time spent initializing.
    func prepare(model: DeepFilterNetModel) throws -> Double

    /// Process a single hop of audio (hopSize samples).
    /// Returns enhanced output samples, or empty if buffering (startup).
    func processHop(_ samples: [Float]) throws -> [Float]

    /// Flush any remaining buffered audio.
    func flush() throws -> [Float]

    /// Reset state for a fresh run.
    func reset()

    /// Whether this engine supports offline (batch) processing.
    var supportsOffline: Bool { get }

    /// Offline batch enhancement (optional).
    func enhanceOffline(_ audio: MLXArray, model: DeepFilterNetModel) throws -> MLXArray
}

// Default: no offline support
extension StreamingEngine {
    public var supportsOffline: Bool { false }
    public func enhanceOffline(_ audio: MLXArray, model: DeepFilterNetModel) throws -> MLXArray {
        fatalError("\(engineType.displayName) does not support offline enhancement")
    }
}
