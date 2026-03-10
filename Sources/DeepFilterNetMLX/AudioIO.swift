import AVFoundation
import Foundation
import MLX

public enum AudioIOError: Error, LocalizedError {
    case cannotCreateBuffer
    case cannotCreateFormat
    case cannotReadChannelData
    case cannotResample

    public var errorDescription: String? {
        switch self {
        case .cannotCreateBuffer:
            return "Failed to allocate audio buffer."
        case .cannotCreateFormat:
            return "Failed to create audio format."
        case .cannotReadChannelData:
            return "Failed to read float channel data."
        case .cannotResample:
            return "Audio resampling failed."
        }
    }
}

public enum AudioIO {
    public static func loadMono(url: URL, targetSampleRate: Int? = nil) throws -> (Int, MLXArray) {
        let file = try AVAudioFile(forReading: url)
        let format = file.processingFormat
        let frameCount = AVAudioFrameCount(file.length)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioIOError.cannotCreateBuffer
        }
        try file.read(into: buffer)

        guard let channels = buffer.floatChannelData else {
            throw AudioIOError.cannotReadChannelData
        }

        let samples = Array(
            UnsafeBufferPointer(start: channels[0], count: Int(buffer.frameLength))
        )
        let sr = Int(format.sampleRate)

        if let targetSampleRate, targetSampleRate > 0, targetSampleRate != sr {
            let resampled = try resample(samples, from: sr, to: targetSampleRate)
            return (targetSampleRate, MLXArray(resampled))
        }
        return (sr, MLXArray(samples))
    }

    public static func saveMonoWav(samples: MLXArray, sampleRate: Double, url: URL) throws {
        let data = samples.asType(.float32).asArray(Float.self)
        guard let format = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: sampleRate,
            channels: 1,
            interleaved: false
        ) else {
            throw AudioIOError.cannotCreateFormat
        }

        let frameCount = AVAudioFrameCount(data.count)
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frameCount) else {
            throw AudioIOError.cannotCreateBuffer
        }
        buffer.frameLength = frameCount

        guard let outputChannel = buffer.floatChannelData?[0] else {
            throw AudioIOError.cannotReadChannelData
        }

        data.withUnsafeBufferPointer { ptr in
            guard let src = ptr.baseAddress else { return }
            memcpy(outputChannel, src, data.count * MemoryLayout<Float>.size)
        }

        let file = try AVAudioFile(
            forWriting: url,
            settings: format.settings,
            commonFormat: .pcmFormatFloat32,
            interleaved: false
        )
        try file.write(from: buffer)
    }

    public static func resample(_ input: [Float], from sourceSampleRate: Int, to targetSampleRate: Int) throws -> [Float] {
        if input.isEmpty || sourceSampleRate == targetSampleRate {
            return input
        }

        guard let inFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(sourceSampleRate),
            channels: 1,
            interleaved: false
        ), let outFormat = AVAudioFormat(
            commonFormat: .pcmFormatFloat32,
            sampleRate: Double(targetSampleRate),
            channels: 1,
            interleaved: false
        ), let converter = AVAudioConverter(from: inFormat, to: outFormat)
        else {
            throw AudioIOError.cannotResample
        }

        let inFrameCount = AVAudioFrameCount(input.count)
        guard let inputBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: inFrameCount) else {
            throw AudioIOError.cannotCreateBuffer
        }
        inputBuffer.frameLength = inFrameCount

        input.withUnsafeBufferPointer { ptr in
            guard let base = ptr.baseAddress else { return }
            memcpy(inputBuffer.floatChannelData![0], base, input.count * MemoryLayout<Float>.size)
        }

        let ratio = Double(targetSampleRate) / Double(sourceSampleRate)
        let estimatedFrames = max(1, Int(ceil(Double(input.count) * ratio)) + 64)
        guard let outputBuffer = AVAudioPCMBuffer(
            pcmFormat: outFormat,
            frameCapacity: AVAudioFrameCount(estimatedFrames)
        ) else {
            throw AudioIOError.cannotCreateBuffer
        }

        final class InputProvider: @unchecked Sendable {
            let buffer: AVAudioPCMBuffer
            var consumed = false

            init(buffer: AVAudioPCMBuffer) {
                self.buffer = buffer
            }
        }

        let provider = InputProvider(buffer: inputBuffer)
        var conversionError: NSError?
        let status = converter.convert(to: outputBuffer, error: &conversionError) { _, outStatus in
            if provider.consumed {
                outStatus.pointee = .endOfStream
                return nil
            }
            provider.consumed = true
            outStatus.pointee = .haveData
            return provider.buffer
        }

        if conversionError != nil {
            throw AudioIOError.cannotResample
        }
        guard status == .haveData || status == .endOfStream || status == .inputRanDry else {
            throw AudioIOError.cannotResample
        }

        let outCount = Int(outputBuffer.frameLength)
        guard let outData = outputBuffer.floatChannelData?[0], outCount > 0 else {
            return []
        }

        return Array(UnsafeBufferPointer(start: outData, count: outCount))
    }
}
