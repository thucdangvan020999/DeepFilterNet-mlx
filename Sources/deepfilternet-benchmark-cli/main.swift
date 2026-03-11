import ArgumentParser
import DeepFilterNetBenchmark
import DeepFilterNetMLX
import Foundation
import MLX

@main
struct BenchmarkCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "deepfilternet-benchmark",
        abstract: "Benchmark DeepFilterNet streaming engines side by side"
    )

    @Argument(help: "Input audio file for benchmarking")
    var input: String

    @Option(name: .shortAndLong, help: "Model repo ID or local directory")
    var model: String = DeepFilterNetModel.defaultRepo

    @Option(name: .long, help: "HuggingFace token")
    var hfToken: String?

    @Option(name: .shortAndLong, help: "Output directory for results")
    var outputDir: String = "benchmark_results"

    @Option(name: .long, help: "Number of warmup runs")
    var warmup: Int = 1

    @Option(name: .long, help: "Number of measurement runs")
    var runs: Int = 3

    @Option(name: .long, help: "Engines to benchmark (comma-separated: mlx,cpu,hybrid,coreml or 'all')")
    var engines: String = "all"

    @Flag(name: .long, help: "Skip offline benchmarks")
    var skipOffline: Bool = false

    @Flag(name: .long, help: "Skip streaming benchmarks")
    var skipStreaming: Bool = false

    @Flag(name: .long, help: "Verbose output")
    var verbose: Bool = false

    @Flag(name: .long, help: "Save streaming output audio for quality comparison")
    var saveOutput: Bool = false

    mutating func run() async throws {
        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input file not found: \(inputURL.path)")
        }

        // Create output directory
        let outputDirURL = URL(fileURLWithPath: outputDir).standardizedFileURL
        try FileManager.default.createDirectory(at: outputDirURL, withIntermediateDirectories: true)

        print("=== DeepFilterNet Streaming Engine Benchmark ===")
        print()

        // Load model
        print("Loading model: \(model)")
        let modelRuntime = try await DeepFilterNetModel.fromPretrained(
            model, hfToken: hfToken, precision: .fp32
        )
        print("Model version: \(modelRuntime.modelVersion)")
        print()

        // Load audio
        print("Loading audio: \(inputURL.path)")
        let (sampleRate, audioMLX) = try AudioIO.loadMono(url: inputURL)
        var audioSamples = audioMLX.asType(.float32).asArray(Float.self)

        if sampleRate != modelRuntime.sampleRate {
            print("Resampling from \(sampleRate)Hz to \(modelRuntime.sampleRate)Hz...")
            audioSamples = try AudioIO.resample(audioSamples, from: sampleRate, to: modelRuntime.sampleRate)
        }

        let audioLength = Double(audioSamples.count) / Double(modelRuntime.sampleRate)
        let totalHops = audioSamples.count / modelRuntime.config.hopSize
        print(String(format: "Audio: %.1fs (%d samples, %d hops)", audioLength, audioSamples.count, totalHops))
        print()

        // Determine which engines to run
        let engineNames = engines.lowercased() == "all"
            ? ["mlx", "cpu", "hybrid", "coreml"]
            : engines.lowercased().split(separator: ",").map(String.init)

        var engineInstances = [(String, StreamingEngine)]()
        for name in engineNames {
            switch name.trimmingCharacters(in: .whitespaces) {
            case "mlx":
                engineInstances.append(("MLX GPU", MLXStreamingEngine()))
            case "cpu":
                engineInstances.append(("CPU + Accelerate", CPUAccelerateEngine()))
            case "hybrid":
                engineInstances.append(("Hybrid", HybridStreamingEngine()))
            case "coreml":
                engineInstances.append(("CoreML + ANE", CoreMLStreamingEngine()))
            default:
                print("Unknown engine: \(name), skipping")
            }
        }

        let runner = BenchmarkRunner(
            model: modelRuntime,
            audioSamples: audioSamples,
            sampleRate: modelRuntime.sampleRate
        )

        // Run benchmarks
        var offlineResults = [OfflineBenchmarkResult]()
        var streamingResults = [StreamingBenchmarkResult]()

        for (displayName, engine) in engineInstances {
            print("--- \(displayName) ---")

            if !skipOffline {
                do {
                    if let result = try runner.benchmarkOffline(
                        engine: engine,
                        warmupRuns: warmup,
                        measureRuns: runs,
                        verbose: verbose
                    ) {
                        offlineResults.append(result)
                    }
                } catch {
                    print("  Offline failed: \(error)")
                }
            }

            if !skipStreaming {
                do {
                    let result = try runner.benchmarkStreaming(
                        engine: engine,
                        warmupRuns: warmup,
                        measureRuns: runs,
                        verbose: verbose
                    )
                    streamingResults.append(result)
                } catch {
                    print("  Streaming failed: \(error)")
                }
            }

            print()
        }

        // Save output audio if requested
        if saveOutput {
            print("Saving streaming output for quality comparison...")
            for (displayName, engine) in engineInstances {
                engine.reset()
                _ = try engine.prepare(model: modelRuntime)
                var outputSamples = [Float]()
                var offset = 0
                while offset + modelRuntime.config.hopSize <= audioSamples.count {
                    let hop = Array(audioSamples[offset..<(offset + modelRuntime.config.hopSize)])
                    let out = try engine.processHop(hop)
                    outputSamples.append(contentsOf: out)
                    offset += modelRuntime.config.hopSize
                }
                let tail = try engine.flush()
                outputSamples.append(contentsOf: tail)

                let safeName = displayName.lowercased()
                    .replacingOccurrences(of: " ", with: "_")
                    .replacingOccurrences(of: "+", with: "")
                    .replacingOccurrences(of: "__", with: "_")
                let outURL = outputDirURL.appendingPathComponent("output_\(safeName).wav")
                let outMLX = MLXArray(outputSamples)
                try AudioIO.saveMonoWav(samples: outMLX, sampleRate: Double(modelRuntime.sampleRate), url: outURL)
                print("  \(displayName) -> \(outURL.path) (\(outputSamples.count) samples)")
            }
            print()
        }

        // Generate timestamp
        let formatter = DateFormatter()
        formatter.dateFormat = "yyyy-MM-dd_HH-mm-ss"
        let timestamp = formatter.string(from: Date())

        // Create suite
        let suite = BenchmarkSuite(
            timestamp: timestamp,
            audioFile: inputURL.lastPathComponent,
            audioLengthSeconds: audioLength,
            sampleRate: modelRuntime.sampleRate,
            modelVersion: modelRuntime.modelVersion,
            offlineResults: offlineResults,
            streamingResults: streamingResults
        )

        // Save JSON results
        let jsonURL = outputDirURL.appendingPathComponent("benchmark_\(timestamp).json")
        let jsonData = try suite.toJSON()
        try jsonData.write(to: jsonURL)
        print("Results saved to: \(jsonURL.path)")

        // Print summary
        printSummary(offlineResults: offlineResults, streamingResults: streamingResults)

        print()
        print("To generate graphs, run:")
        print("  python Scripts/generate_benchmark_graphs.py \(jsonURL.path)")
    }

    func printSummary(offlineResults: [OfflineBenchmarkResult], streamingResults: [StreamingBenchmarkResult]) {
        print()
        print("=== SUMMARY ===")
        print()

        if !offlineResults.isEmpty {
            print("Offline Enhancement:")
            for r in offlineResults {
                let time = String(format: "%.3f", r.totalTimeSeconds)
                let rt = String(format: "%.1f", r.realtimeFactor)
                print("  \(r.engineType.displayName): \(time)s (\(rt)x RT)")
            }
            print()
        }

        if !streamingResults.isEmpty {
            print("Streaming Enhancement:")
            for r in streamingResults {
                let initT = String(format: "%.3f", r.initTimeSeconds)
                let startT = String(format: "%.3f", r.startupPerHopMs)
                let steadyT = String(format: "%.3f", r.steadyStatePerHopMs)
                let totalT = String(format: "%.3f", r.totalStreamingTimeSeconds)
                let rt = String(format: "%.1f", r.realtimeFactor)
                print("  \(r.engineType.displayName):")
                print("    Init: \(initT)s | Startup: \(startT)ms/hop | Steady: \(steadyT)ms/hop")
                print("    Total: \(totalT)s | RT: \(rt)x")
            }
            print()

            if let fastest = streamingResults.min(by: { $0.steadyStatePerHopSeconds < $1.steadyStatePerHopSeconds }) {
                let ms = String(format: "%.3f", fastest.steadyStatePerHopMs)
                print("Lowest steady-state latency: \(fastest.engineType.displayName) (\(ms)ms/hop)")
            }
            if let bestRT = streamingResults.max(by: { $0.realtimeFactor < $1.realtimeFactor }) {
                let rt = String(format: "%.1f", bestRT.realtimeFactor)
                print("Best overall throughput: \(bestRT.engineType.displayName) (\(rt)x realtime)")
            }
        }
    }
}
