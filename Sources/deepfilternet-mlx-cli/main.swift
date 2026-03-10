import ArgumentParser
import DeepFilterNetMLX
import Foundation
import MLX

@main
struct DeepFilterNetCLI: AsyncParsableCommand {
    static let configuration = CommandConfiguration(
        commandName: "deepfilternet-mlx",
        abstract: "Standalone Swift/MLX DeepFilterNet with optional fused Metal kernels"
    )

    @Argument(help: "Input audio file")
    var input: String

    @Option(name: .shortAndLong, help: "Output WAV path")
    var output: String?

    @Option(name: .shortAndLong, help: "Model repo ID or local model directory")
    var model: String = DeepFilterNetModel.defaultRepo

    @Option(name: .long, help: "HuggingFace token")
    var hfToken: String?

    @Flag(name: .long, help: "Use streaming mode")
    var stream: Bool = false

    @Option(name: .long, help: "Streaming chunk size in ms")
    var chunkMs: Double = 10.0

    @Option(name: .long, help: "Streaming flush padding (frames)")
    var padEndFrames: Int = 3

    @Flag(name: .long, help: "Disable delay compensation in streaming mode")
    var noDelayCompensation: Bool = false

    @Option(name: .long, help: "Materialize MLX graph state every N hops (lower reduces long-graph overhead)")
    var materializeEveryHops: Int = 96

    @Flag(name: .long, help: "Enable stage skipping based on local SNR thresholds")
    var stageSkipping: Bool = false

    @Option(name: .long, help: "Min dB local SNR threshold for stage skipping")
    var minDbThresh: Float = -10.0

    @Option(name: .long, help: "Max dB local SNR threshold for running ERB stage")
    var maxDbErbThresh: Float = 30.0

    @Option(name: .long, help: "Max dB local SNR threshold for running DF stage")
    var maxDbDfThresh: Float = 20.0

    @Flag(name: .long, help: "Enable detailed per-stage streaming profiling output")
    var profileStream: Bool = false

    @Option(name: .long, help: "Performance preset: throughput or safe")
    var performance: String = "throughput"

    @Option(name: .long, help: "Inference precision: fp32 or fp16")
    var precision: String = "fp32"

    mutating func run() async throws {
        let inputURL = URL(fileURLWithPath: input).standardizedFileURL
        guard FileManager.default.fileExists(atPath: inputURL.path) else {
            throw ValidationError("Input file not found: \(inputURL.path)")
        }

        let outputURL: URL = {
            if let output {
                return URL(fileURLWithPath: output).standardizedFileURL
            }
            let stem = inputURL.deletingPathExtension().lastPathComponent
            let parent = inputURL.deletingLastPathComponent()
            return parent.appendingPathComponent("\(stem)_enhanced_mlx.wav")
        }()

        let started = CFAbsoluteTimeGetCurrent()
        guard let runtimePrecision = DeepFilterNetPrecision(rawValue: precision.lowercased()) else {
            throw ValidationError("Unsupported precision: \(precision)")
        }
        let modelRuntime = try await DeepFilterNetModel.fromPretrained(
            model,
            hfToken: hfToken,
            precision: runtimePrecision
        )

        switch performance.lowercased() {
        case "safe":
            modelRuntime.configurePerformance(.safe)
        case "throughput":
            modelRuntime.configurePerformance(.throughput)
        default:
            throw ValidationError("Unsupported performance preset: \(performance)")
        }

        let (sampleRate, audio) = try AudioIO.loadMono(url: inputURL)

        let inputAudio: MLXArray
        if sampleRate != modelRuntime.sampleRate {
            let resampled = try AudioIO.resample(
                audio.asArray(Float.self),
                from: sampleRate,
                to: modelRuntime.sampleRate
            )
            inputAudio = MLXArray(resampled)
        } else {
            inputAudio = audio
        }

        let enhanced: MLXArray
        if stream {
            let chunkSamples = max(modelRuntime.config.hopSize, Int(Double(modelRuntime.sampleRate) * chunkMs / 1000.0))
            let streamConfig = DeepFilterNetStreamingConfig(
                padEndFrames: max(0, padEndFrames),
                compensateDelay: !noDelayCompensation,
                enableStageSkipping: stageSkipping,
                minDbThresh: minDbThresh,
                maxDbErbThresh: maxDbErbThresh,
                maxDbDfThresh: maxDbDfThresh,
                enableProfiling: profileStream,
                profilingForceEvalPerStage: profileStream,
                materializeEveryHops: max(1, materializeEveryHops)
            )

            if profileStream {
                let streamer = modelRuntime.createStreamer(config: streamConfig)
                let samples = inputAudio.asType(.float32)
                var outputChunks = [MLXArray]()
                outputChunks.reserveCapacity(max(1, samples.shape[0] / chunkSamples))

                var start = 0
                while start < samples.shape[0] {
                    let end = min(start + chunkSamples, samples.shape[0])
                    let out = try streamer.processChunk(samples[start..<end])
                    if out.shape[0] > 0 {
                        outputChunks.append(out)
                    }
                    start = end
                }
                let tail = try streamer.flushMLX()
                if tail.shape[0] > 0 {
                    outputChunks.append(tail)
                }
                if outputChunks.isEmpty {
                    enhanced = MLXArray.zeros([0], type: Float.self)
                } else {
                    enhanced = MLX.clip(MLX.concatenated(outputChunks, axis: 0), min: -1.0, max: 1.0)
                }
                if let summary = streamer.profilingSummary() {
                    print(summary)
                }
            } else {
                enhanced = try modelRuntime.enhanceStreaming(
                    inputAudio,
                    chunkSamples: chunkSamples,
                    config: streamConfig
                )
            }
        } else {
            enhanced = try modelRuntime.enhance(inputAudio)
        }

        try AudioIO.saveMonoWav(
            samples: enhanced,
            sampleRate: Double(modelRuntime.sampleRate),
            url: outputURL
        )

        let elapsed = CFAbsoluteTimeGetCurrent() - started
        print("Input : \(inputURL.path)")
        print("Output: \(outputURL.path)")
        print("Model : \(modelRuntime.modelVersion)")
        print("Mode  : \(stream ? "streaming" : "offline")")
        print("Prec  : \(modelRuntime.precision.rawValue)")
        print(String(format: "Time  : %.3fs", elapsed))
    }
}
