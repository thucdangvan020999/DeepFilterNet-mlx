import DeepFilterNetMLX
import Foundation
import MLX

// MARK: - Benchmark Runner

public final class BenchmarkRunner {
    public let model: DeepFilterNetModel
    public let audioSamples: [Float]
    public let sampleRate: Int
    public let hopSize: Int

    public var audioLengthSeconds: Double {
        Double(audioSamples.count) / Double(sampleRate)
    }

    public init(model: DeepFilterNetModel, audioSamples: [Float], sampleRate: Int) {
        self.model = model
        self.audioSamples = audioSamples
        self.sampleRate = sampleRate
        self.hopSize = model.config.hopSize
    }

    // MARK: - Streaming Benchmark

    public func benchmarkStreaming(
        engine: StreamingEngine,
        warmupRuns: Int = 1,
        measureRuns: Int = 3,
        verbose: Bool = true
    ) throws -> StreamingBenchmarkResult {
        if verbose {
            print("  [\(engine.engineType.displayName)] Preparing engine...")
        }

        // Prepare (load weights, build state)
        let initTime = try engine.prepare(model: model)
        if verbose {
            print(String(format: "  [\(engine.engineType.displayName)] Init: %.3fs", initTime))
        }

        // Warmup runs
        for w in 0..<warmupRuns {
            if verbose {
                print("  [\(engine.engineType.displayName)] Warmup \(w + 1)/\(warmupRuns)...")
            }
            engine.reset()
            _ = try runStreamingPass(engine: engine)
        }

        // Measurement runs
        var allTimings = [[HopTiming]]()
        var allTotals = [Double]()

        for r in 0..<measureRuns {
            if verbose {
                print("  [\(engine.engineType.displayName)] Measure \(r + 1)/\(measureRuns)...")
            }
            engine.reset()
            let (timings, totalTime) = try runStreamingPassTimed(engine: engine)
            allTimings.append(timings)
            allTotals.append(totalTime)
        }

        // Use the median run
        let medianIdx = allTotals.enumerated().sorted { $0.element < $1.element }[measureRuns / 2].offset
        let bestTimings = allTimings[medianIdx]
        let medianTotal = allTotals[medianIdx]

        // Calculate startup time (first 2 output-producing hops)
        let startupHops = bestTimings.prefix(min(2, bestTimings.count))
        let startupTime = startupHops.reduce(0.0) { $0 + $1.durationSeconds }

        // Calculate steady-state (median of hops after the first 2)
        let steadyHops = Array(bestTimings.dropFirst(2))
        let steadyPerHop: Double
        if steadyHops.isEmpty {
            steadyPerHop = startupTime / max(1, Double(startupHops.count))
        } else {
            let sorted = steadyHops.map(\.durationSeconds).sorted()
            steadyPerHop = sorted[sorted.count / 2]
        }

        let result = StreamingBenchmarkResult(
            engineType: engine.engineType,
            audioLengthSeconds: audioLengthSeconds,
            totalHops: bestTimings.count,
            initTimeSeconds: initTime,
            startupTimeSeconds: startupTime,
            steadyStatePerHopSeconds: steadyPerHop,
            totalStreamingTimeSeconds: medianTotal,
            perHopTimings: bestTimings
        )

        if verbose {
            print(String(format: "  [\(engine.engineType.displayName)] Total: %.3fs, Startup: %.3fms/hop, Steady: %.3fms/hop, RT: %.1fx",
                         result.totalStreamingTimeSeconds,
                         result.startupPerHopMs,
                         result.steadyStatePerHopMs,
                         result.realtimeFactor))
        }

        return result
    }

    // MARK: - Offline Benchmark

    public func benchmarkOffline(
        engine: StreamingEngine,
        warmupRuns: Int = 1,
        measureRuns: Int = 3,
        verbose: Bool = true
    ) throws -> OfflineBenchmarkResult? {
        guard engine.supportsOffline else {
            if verbose {
                print("  [\(engine.engineType.displayName)] Skipping offline (not supported)")
            }
            return nil
        }

        // Prepare engine (load weights, compile model, etc.)
        let initTime = try engine.prepare(model: model)
        if verbose {
            print(String(format: "  [\(engine.engineType.displayName)] Offline init: %.3fs", initTime))
        }

        let audio = MLXArray(audioSamples)

        // Warmup
        for w in 0..<warmupRuns {
            if verbose {
                print("  [\(engine.engineType.displayName)] Offline warmup \(w + 1)/\(warmupRuns)...")
            }
            let result = try engine.enhanceOffline(audio, model: model)
            eval(result)
        }

        // Measure
        var times = [Double]()
        for r in 0..<measureRuns {
            if verbose {
                print("  [\(engine.engineType.displayName)] Offline measure \(r + 1)/\(measureRuns)...")
            }
            let t0 = CFAbsoluteTimeGetCurrent()
            let result = try engine.enhanceOffline(audio, model: model)
            eval(result)
            times.append(CFAbsoluteTimeGetCurrent() - t0)
        }

        let medianTime = times.sorted()[measureRuns / 2]
        let result = OfflineBenchmarkResult(
            engineType: engine.engineType,
            audioLengthSeconds: audioLengthSeconds,
            totalTimeSeconds: medianTime
        )

        if verbose {
            print(String(format: "  [\(engine.engineType.displayName)] Offline: %.3fs (%.1fx RT)",
                         result.totalTimeSeconds, result.realtimeFactor))
        }

        return result
    }

    // MARK: - Helpers

    /// Run streaming pass, discard output, return output sample count
    private func runStreamingPass(engine: StreamingEngine) throws -> Int {
        var offset = 0
        var totalOut = 0
        while offset + hopSize <= audioSamples.count {
            let hop = Array(audioSamples[offset..<(offset + hopSize)])
            let out = try engine.processHop(hop)
            totalOut += out.count
            offset += hopSize
        }
        let tail = try engine.flush()
        totalOut += tail.count
        return totalOut
    }

    /// Run streaming pass with per-hop timing
    private func runStreamingPassTimed(engine: StreamingEngine) throws -> ([HopTiming], Double) {
        var timings = [HopTiming]()
        var offset = 0
        var hopIdx = 0

        let totalStart = CFAbsoluteTimeGetCurrent()

        while offset + hopSize <= audioSamples.count {
            let hop = Array(audioSamples[offset..<(offset + hopSize)])
            let t0 = CFAbsoluteTimeGetCurrent()
            let out = try engine.processHop(hop)
            let dt = CFAbsoluteTimeGetCurrent() - t0

            // Only record hops that produce output (skip initial buffering)
            if !out.isEmpty {
                timings.append(HopTiming(hopIndex: hopIdx, durationSeconds: dt))
                hopIdx += 1
            }
            offset += hopSize
        }

        let t0 = CFAbsoluteTimeGetCurrent()
        _ = try engine.flush()
        let flushDt = CFAbsoluteTimeGetCurrent() - t0
        if flushDt > 0.0001 {
            timings.append(HopTiming(hopIndex: hopIdx, durationSeconds: flushDt))
        }

        let totalTime = CFAbsoluteTimeGetCurrent() - totalStart
        return (timings, totalTime)
    }
}
