import Foundation
import HuggingFace
import MLX
import MLXNN

extension DeepFilterNetModel {
    // MARK: - Feature Helpers

    func bandMeanNorm(_ x: MLXArray) -> MLXArray {
        let frames = x.shape[0]
        let a = normAlpha()
        let oneMinusA = Float(1.0) - a
        let time = MLXArray.arange(frames).asType(.float32)
        let powers = MLX.pow(MLXArray(a), time) // [T]
        let invPowers = MLXArray(Float(1.0)) / powers

        let scaled = x * invPowers.expandedDimensions(axis: 1)
        let accum = cumsum(scaled, axis: 0)

        let initState = MLXArray(Self.linspace(start: -60.0, end: -90.0, count: x.shape[1]))
            .expandedDimensions(axis: 0)
        let state = powers.expandedDimensions(axis: 1) * (initState + MLXArray(oneMinusA) * accum)
        return (x - state) / MLXArray(Float(40.0))
    }

    func bandUnitNorm(real: MLXArray, imag: MLXArray) -> (MLXArray, MLXArray) {
        let frames = real.shape[0]
        let a = normAlpha()
        let oneMinusA = Float(1.0) - a
        let time = MLXArray.arange(frames).asType(.float32)
        let powers = MLX.pow(MLXArray(a), time) // [T]
        let invPowers = MLXArray(Float(1.0)) / powers

        let mag = MLX.sqrt(real.square() + imag.square())
        let scaled = mag * invPowers.expandedDimensions(axis: 1)
        let accum = cumsum(scaled, axis: 0)

        let initState = MLXArray(Self.linspace(start: 0.001, end: 0.0001, count: real.shape[1]))
            .expandedDimensions(axis: 0)
        let state = powers.expandedDimensions(axis: 1) * (initState + MLXArray(oneMinusA) * accum)
        let denom = MLX.sqrt(MLX.maximum(state, MLXArray(Float(1e-12))))
        return (real / denom, imag / denom)
    }

    // Exact sequential EMA path (libDF-style), primarily for DF1 parity.
    func bandMeanNormExact(_ x: MLXArray) -> MLXArray {
        let frames = x.shape[0]
        let bands = x.shape[1]
        let a = normAlpha()
        let oneMinusA = Float(1.0) - a

        let xVals = x.asArray(Float.self)
        var out = Array<Float>(repeating: 0, count: xVals.count)
        var state = Self.linspace(start: -60.0, end: -90.0, count: bands)

        for t in 0..<frames {
            let base = t * bands
            for e in 0..<bands {
                let idx = base + e
                let xv = xVals[idx]
                state[e] = xv * oneMinusA + state[e] * a
                out[idx] = (xv - state[e]) / 40.0
            }
        }
        return MLXArray(out).reshaped([frames, bands])
    }

    // Exact sequential complex unit-norm path (libDF-style), primarily for DF1 parity.
    func bandUnitNormExact(real: MLXArray, imag: MLXArray) -> (MLXArray, MLXArray) {
        let frames = real.shape[0]
        let freqs = real.shape[1]
        let a = normAlpha()
        let oneMinusA = Float(1.0) - a

        let rVals = real.asArray(Float.self)
        let iVals = imag.asArray(Float.self)
        var outR = Array<Float>(repeating: 0, count: rVals.count)
        var outI = Array<Float>(repeating: 0, count: iVals.count)
        var state = Self.linspace(start: 0.001, end: 0.0001, count: freqs)

        for t in 0..<frames {
            let base = t * freqs
            for f in 0..<freqs {
                let idx = base + f
                let rr = rVals[idx]
                let ii = iVals[idx]
                let mag = (rr * rr + ii * ii).squareRoot()
                state[f] = mag * oneMinusA + state[f] * a
                let den = state[f].squareRoot()
                outR[idx] = rr / den
                outI[idx] = ii / den
            }
        }

        return (
            MLXArray(outR).reshaped([frames, freqs]),
            MLXArray(outI).reshaped([frames, freqs])
        )
    }

    func normAlpha() -> Float {
        normAlphaValue
    }

    static func computeNormAlpha(hopSize: Int, sampleRate: Int) -> Float {
        let aRaw = exp(-Float(hopSize) / Float(sampleRate))
        var precision = 3
        var a: Float = 1.0
        while a >= 1.0 {
            let scale = powf(10, Float(precision))
            a = (aRaw * scale).rounded() / scale
            precision += 1
        }
        return a
    }

    func applyLookahead(feature: MLXArray, lookahead: Int) -> MLXArray {
        guard lookahead > 0 else { return feature }
        let t = feature.shape[2]
        guard t > lookahead else { return feature }
        let shifted = feature[0..., 0..., lookahead..<t, 0...]
        let pad = MLXArray.zeros([feature.shape[0], feature.shape[1], lookahead, feature.shape[3]], type: Float.self)
        return MLX.concatenated([shifted, pad], axis: 2)
    }

}
