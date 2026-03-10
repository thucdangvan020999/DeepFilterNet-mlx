import Foundation
import MLX

public enum DeepFilterNetDSP {
    public static func stft(
        audio: MLXArray,
        fftLen: Int,
        hopLength: Int,
        winLen: Int,
        window: MLXArray,
        center: Bool = false
    ) -> MLXArray {
        guard fftLen > 0, hopLength > 0, winLen > 0 else {
            return MLXArray.zeros([0, 0], dtype: .complex64)
        }

        var signal = audio
        if center {
            let padAmount = fftLen / 2
            if padAmount > 0 {
                let pad = MLXArray.zeros([padAmount], type: Float.self)
                signal = MLX.concatenated([pad, signal, pad], axis: 0)
            }
        }

        let signalLen = signal.shape[0]
        guard signalLen >= winLen else {
            return MLXArray.zeros([0, fftLen / 2 + 1], dtype: .complex64)
        }

        let numFrames = 1 + (signalLen - winLen + hopLength - 1) / hopLength
        guard numFrames > 0 else {
            return MLXArray.zeros([0, fftLen / 2 + 1], dtype: .complex64)
        }

        let requiredLen = winLen + (numFrames - 1) * hopLength
        if signalLen < requiredLen {
            let tail = MLXArray.zeros([requiredLen - signalLen], type: Float.self)
            signal = MLX.concatenated([signal, tail], axis: 0)
        }

        var frames = asStrided(signal, [numFrames, winLen], strides: [hopLength, 1], offset: 0)
        frames = frames * adjustedWindow(window, targetLength: winLen)

        if winLen < fftLen {
            let rightPad = MLXArray.zeros([numFrames, fftLen - winLen], type: Float.self)
            frames = MLX.concatenated([frames, rightPad], axis: 1)
        } else if winLen > fftLen {
            frames = frames[0..<numFrames, 0..<fftLen]
        }

        return MLXFFT.rfft(frames, axis: 1)
    }

    public static func istft(
        real: MLXArray,
        imag: MLXArray,
        fftLen: Int,
        hopLength: Int,
        winLen: Int,
        window: MLXArray,
        center: Bool = false,
        audioLength: Int? = nil
    ) -> MLXArray {
        guard fftLen > 0, hopLength > 0, winLen > 0 else {
            return MLXArray.zeros([0], type: Float.self)
        }
        guard real.ndim == 3, imag.ndim == 3, real.shape == imag.shape else {
            return MLXArray.zeros([0], type: Float.self)
        }
        guard real.shape[0] == 1, real.shape[0] > 0 else {
            return MLXArray.zeros([0], type: Float.self)
        }

        let realT = real[0].transposed(1, 0)
        let imagT = imag[0].transposed(1, 0)
        let complexSpec = realT + MLXArray(real: Float(0), imaginary: Float(1)) * imagT

        var frames = MLXFFT.irfft(complexSpec, axis: 1)
        let numFrames = frames.shape[0]
        guard numFrames > 0 else {
            return MLXArray.zeros([0], type: Float.self)
        }

        let frameWidth = Swift.min(winLen, frames.shape[1])
        guard frameWidth > 0 else {
            return MLXArray.zeros([0], type: Float.self)
        }

        frames = frames[0..<numFrames, 0..<frameWidth]
        let synthesisWindow = adjustedWindow(window, targetLength: frameWidth)
        let windowedFrames = frames * synthesisWindow

        let fullLength = (numFrames - 1) * hopLength + frameWidth
        guard fullLength > 0 else {
            return MLXArray.zeros([0], type: Float.self)
        }

        let frameOffsets = MLXArray.arange(numFrames).expandedDimensions(axis: 1)
            * MLXArray(Int32(hopLength))
        let sampleOffsets = MLXArray.arange(frameWidth).expandedDimensions(axis: 0)
        let indices = (frameOffsets + sampleOffsets).reshaped(-1)

        let flatFrames = windowedFrames.reshaped(-1)
        var output = MLXArray.zeros([fullLength], type: Float.self)
        output = output.at[indices].add(flatFrames)

        let windowSq = synthesisWindow * synthesisWindow
        let tiledWindowSq = MLX.repeated(
            windowSq.expandedDimensions(axis: 0), count: numFrames, axis: 0
        ).reshaped(-1)

        var windowSum = MLXArray.zeros([fullLength], type: Float.self)
        windowSum = windowSum.at[indices].add(tiledWindowSq)

        let eps = MLXArray(Float(1e-8))
        var result = output / MLX.maximum(windowSum, eps)

        if center {
            let trim = fftLen / 2
            if fullLength > 2 * trim {
                result = result[trim..<(fullLength - trim)]
            }
        }
        if let audioLength, result.shape[0] > audioLength {
            result = result[0..<audioLength]
        }

        return result
    }

    private static func adjustedWindow(_ window: MLXArray, targetLength: Int) -> MLXArray {
        guard targetLength > 0 else {
            return MLXArray.zeros([0], type: Float.self)
        }
        if window.shape[0] == targetLength {
            return window
        }
        if window.shape[0] > targetLength {
            return window[0..<targetLength]
        }
        let pad = MLXArray.zeros([targetLength - window.shape[0]], type: Float.self)
        return MLX.concatenated([window, pad], axis: 0)
    }
}
