import Foundation
import MLX

public extension DeepFilterNetModel {
    func enhanceFile(
        inputURL: URL,
        outputURL: URL,
        useStreaming: Bool = false,
        chunkMilliseconds: Double = 10.0,
        streamingConfig: DeepFilterNetStreamingConfig = DeepFilterNetStreamingConfig()
    ) throws {
        let (inputSampleRate, inputAudio) = try AudioIO.loadMono(url: inputURL)

        let modelInput: MLXArray
        if inputSampleRate != sampleRate {
            let resampled = try AudioIO.resample(
                inputAudio.asArray(Float.self),
                from: inputSampleRate,
                to: sampleRate
            )
            modelInput = MLXArray(resampled)
        } else {
            modelInput = inputAudio
        }

        let enhanced: MLXArray
        if useStreaming {
            let chunkSamples = max(config.hopSize, Int(Double(sampleRate) * chunkMilliseconds / 1000.0))
            enhanced = try enhanceStreaming(
                modelInput,
                chunkSamples: chunkSamples,
                config: streamingConfig
            )
        } else {
            enhanced = try enhance(modelInput)
        }

        try AudioIO.saveMonoWav(
            samples: enhanced,
            sampleRate: Double(sampleRate),
            url: outputURL
        )
    }
}
