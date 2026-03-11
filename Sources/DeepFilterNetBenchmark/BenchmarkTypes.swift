import Foundation

// MARK: - Engine Identification

public enum StreamingEngineType: String, CaseIterable, Sendable {
    case mlxGPU = "mlx_gpu"
    case cpuAccelerate = "cpu_accelerate"
    case hybrid = "hybrid_compile_cpu"
    case coreML = "coreml_ane"

    public var displayName: String {
        switch self {
        case .mlxGPU: return "MLX GPU"
        case .cpuAccelerate: return "CPU + Accelerate"
        case .hybrid: return "Hybrid (compile + CPU GRU)"
        case .coreML: return "CoreML + ANE"
        }
    }
}

// MARK: - Benchmark Results

public struct HopTiming: Sendable {
    public let hopIndex: Int
    public let durationSeconds: Double

    public init(hopIndex: Int, durationSeconds: Double) {
        self.hopIndex = hopIndex
        self.durationSeconds = durationSeconds
    }
}

public struct StreamingBenchmarkResult: Sendable {
    public let engineType: StreamingEngineType
    public let audioLengthSeconds: Double
    public let totalHops: Int

    // Timing breakdown
    public let initTimeSeconds: Double
    public let startupTimeSeconds: Double    // First 2 hops
    public let steadyStatePerHopSeconds: Double  // Median of hops 3+
    public let totalStreamingTimeSeconds: Double
    public let perHopTimings: [HopTiming]

    // Derived metrics
    public var realtimeFactor: Double {
        guard totalStreamingTimeSeconds > 0 else { return 0 }
        return audioLengthSeconds / totalStreamingTimeSeconds
    }

    public var startupPerHopMs: Double {
        guard totalHops >= 2 else { return startupTimeSeconds * 1000 }
        return (startupTimeSeconds / 2.0) * 1000.0
    }

    public var steadyStatePerHopMs: Double {
        steadyStatePerHopSeconds * 1000.0
    }

    public init(
        engineType: StreamingEngineType,
        audioLengthSeconds: Double,
        totalHops: Int,
        initTimeSeconds: Double,
        startupTimeSeconds: Double,
        steadyStatePerHopSeconds: Double,
        totalStreamingTimeSeconds: Double,
        perHopTimings: [HopTiming]
    ) {
        self.engineType = engineType
        self.audioLengthSeconds = audioLengthSeconds
        self.totalHops = totalHops
        self.initTimeSeconds = initTimeSeconds
        self.startupTimeSeconds = startupTimeSeconds
        self.steadyStatePerHopSeconds = steadyStatePerHopSeconds
        self.totalStreamingTimeSeconds = totalStreamingTimeSeconds
        self.perHopTimings = perHopTimings
    }
}

public struct OfflineBenchmarkResult: Sendable {
    public let engineType: StreamingEngineType
    public let audioLengthSeconds: Double
    public let totalTimeSeconds: Double

    public var realtimeFactor: Double {
        guard totalTimeSeconds > 0 else { return 0 }
        return audioLengthSeconds / totalTimeSeconds
    }

    public init(
        engineType: StreamingEngineType,
        audioLengthSeconds: Double,
        totalTimeSeconds: Double
    ) {
        self.engineType = engineType
        self.audioLengthSeconds = audioLengthSeconds
        self.totalTimeSeconds = totalTimeSeconds
    }
}

public struct BenchmarkSuite: Sendable {
    public let timestamp: String
    public let audioFile: String
    public let audioLengthSeconds: Double
    public let sampleRate: Int
    public let modelVersion: String
    public let offlineResults: [OfflineBenchmarkResult]
    public let streamingResults: [StreamingBenchmarkResult]

    public init(
        timestamp: String,
        audioFile: String,
        audioLengthSeconds: Double,
        sampleRate: Int,
        modelVersion: String,
        offlineResults: [OfflineBenchmarkResult],
        streamingResults: [StreamingBenchmarkResult]
    ) {
        self.timestamp = timestamp
        self.audioFile = audioFile
        self.audioLengthSeconds = audioLengthSeconds
        self.sampleRate = sampleRate
        self.modelVersion = modelVersion
        self.offlineResults = offlineResults
        self.streamingResults = streamingResults
    }
}

// MARK: - JSON Encoding

extension BenchmarkSuite {
    public func toJSON() throws -> Data {
        var dict: [String: Any] = [
            "timestamp": timestamp,
            "audio_file": audioFile,
            "audio_length_seconds": audioLengthSeconds,
            "sample_rate": sampleRate,
            "model_version": modelVersion,
        ]

        var offlineArr = [[String: Any]]()
        for r in offlineResults {
            offlineArr.append([
                "engine": r.engineType.rawValue,
                "engine_display": r.engineType.displayName,
                "audio_length_seconds": r.audioLengthSeconds,
                "total_time_seconds": r.totalTimeSeconds,
                "realtime_factor": r.realtimeFactor,
            ])
        }
        dict["offline_results"] = offlineArr

        var streamArr = [[String: Any]]()
        for r in streamingResults {
            var entry: [String: Any] = [
                "engine": r.engineType.rawValue,
                "engine_display": r.engineType.displayName,
                "audio_length_seconds": r.audioLengthSeconds,
                "total_hops": r.totalHops,
                "init_time_seconds": r.initTimeSeconds,
                "startup_time_seconds": r.startupTimeSeconds,
                "startup_per_hop_ms": r.startupPerHopMs,
                "steady_state_per_hop_ms": r.steadyStatePerHopMs,
                "total_streaming_time_seconds": r.totalStreamingTimeSeconds,
                "realtime_factor": r.realtimeFactor,
            ]
            let hopTimingsArr = r.perHopTimings.map { ["hop": $0.hopIndex, "duration_ms": $0.durationSeconds * 1000.0] }
            entry["per_hop_timings_ms"] = hopTimingsArr
            streamArr.append(entry)
        }
        dict["streaming_results"] = streamArr

        return try JSONSerialization.data(withJSONObject: dict, options: [.prettyPrinted, .sortedKeys])
    }
}
