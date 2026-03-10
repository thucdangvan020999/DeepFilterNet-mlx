import Foundation
import HuggingFace
import MLX
import MLXNN

extension DeepFilterNetModel {
    // MARK: - Layer Helpers

    func applyEncoderConv(
        _ x: MLXArray,
        prefix: String,
        main: Int,
        pointwise: Int?,
        bn: Int,
        fstride: Int
    ) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weightKey: "\(prefix).\(main).weight",
            bias: nil,
            fstride: fstride,
            lookahead: 0
        )
        if let pointwise {
            y = try conv2dLayer(
                y,
                weightKey: "\(prefix).\(pointwise).weight",
                bias: nil,
                fstride: 1,
                lookahead: 0
            )
        }
        y = try batchNorm(y, prefix: "\(prefix).\(bn)")
        return relu(y)
    }

    func applyPathwayConv(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weightKey: "\(prefix).0.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try batchNorm(y, prefix: "\(prefix).1")
        return relu(y)
    }

    func applyTransposeBlock(_ x: MLXArray, prefix: String, fstride: Int) throws -> MLXArray {
        var y = try convTranspose2dLayer(
            x,
            weightKey: "\(prefix).0.weight",
            fstride: fstride,
            groups: config.convCh
        )
        y = try conv2dLayer(
            y,
            weightKey: "\(prefix).1.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        return try batchNorm(y, prefix: "\(prefix).2")
    }

    func applyRegularBlock(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weightKey: "\(prefix).0.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try conv2dLayer(
            y,
            weightKey: "\(prefix).1.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        return try batchNorm(y, prefix: "\(prefix).2")
    }

    func applyOutputConv(_ x: MLXArray, prefix: String) throws -> MLXArray {
        var y = try conv2dLayer(
            x,
            weightKey: "\(prefix).0.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        y = try batchNorm(y, prefix: "\(prefix).1")
        return y
    }

    func convWeightOHWI(_ key: String) throws -> MLXArray {
        if let cached = conv2dWeightsOHWI[key] {
            return cached
        }
        return try w(key).transposed(0, 2, 3, 1)
    }

    func denseTransposeWeight(_ key: String) throws -> MLXArray? {
        if let dense = convTransposeDenseWeights[key] {
            return dense
        }
        if config.convCh <= 1 {
            return try w(key).transposed(1, 2, 3, 0)
        }
        return nil
    }

    func batchNormAffine(prefix: String) throws -> (MLXArray, MLXArray) {
        if let scale = bnScale[prefix], let bias = bnBias[prefix] {
            return (scale, bias)
        }

        let gamma = try w("\(prefix).weight")
        let beta = try w("\(prefix).bias")
        let mean = try w("\(prefix).running_mean")
        let variance = try w("\(prefix).running_var")
        let scale = (gamma / MLX.sqrt(variance + MLXArray(Float(1e-5))))
            .reshaped([1, gamma.shape[0], 1, 1])
        let shift = (beta - mean * (gamma / MLX.sqrt(variance + MLXArray(Float(1e-5)))))
            .reshaped([1, beta.shape[0], 1, 1])
        return (scale, shift)
    }

    static func conv2dBCHW(
        _ xBCHW: MLXArray,
        weightOHWI: MLXArray,
        bias: MLXArray?,
        fstride: Int,
        lookahead: Int
    ) -> MLXArray {
        let kT = weightOHWI.shape[1]
        let kF = weightOHWI.shape[2]
        let inPerGroup = weightOHWI.shape[3]
        let inChannels = xBCHW.shape[1]
        let groups = max(1, inChannels / max(1, inPerGroup))

        let rawLeft = kT - 1 - lookahead
        let timeCrop = max(0, -rawLeft)
        let timePadLeft = max(0, rawLeft)
        let timePadRight = max(0, lookahead)
        let freqPad = kF / 2

        var x = xBCHW.transposed(0, 2, 3, 1)  // NHWC
        if timeCrop > 0, x.shape[1] > timeCrop {
            x = x[0..., timeCrop..., 0..., 0...]
        }
        x = MLX.padded(
            x,
            widths: [
                .init(0),
                .init((timePadLeft, timePadRight)),
                .init((freqPad, freqPad)),
                .init(0),
            ],
            mode: .constant
        )

        var y = MLX.conv2d(x, weightOHWI, stride: [1, fstride], padding: [0, 0], groups: groups)
        if let bias {
            y = y + bias.reshaped([1, 1, 1, bias.shape[0]])
        }
        return y.transposed(0, 3, 1, 2)
    }

    static func convTranspose2dBCHWDense(
        _ xBCHW: MLXArray,
        denseWeight: MLXArray,
        fstride: Int
    ) -> MLXArray {
        var x = xBCHW.transposed(0, 2, 3, 1)  // NHWC
        let kT = denseWeight.shape[1]
        let kF = denseWeight.shape[2]
        let padding = IntOrPair((kT - 1, kF / 2))
        let outputPadding = IntOrPair((0, kF / 2))
        x = MLX.convTransposed2d(
            x,
            denseWeight,
            stride: [1, fstride],
            padding: padding,
            outputPadding: outputPadding,
            groups: 1
        )
        return x.transposed(0, 3, 1, 2)
    }

    func conv2dLayer(
        _ xBCHW: MLXArray,
        weightKey: String,
        bias: MLXArray?,
        fstride: Int,
        lookahead: Int
    ) throws -> MLXArray {
        if let wOHWI = conv2dWeightsOHWI[weightKey] {
            return conv2dLayer(
                xBCHW,
                weightOHWI: wOHWI,
                bias: bias,
                fstride: fstride,
                lookahead: lookahead
            )
        }
        return try conv2dLayer(
            xBCHW,
            weight: w(weightKey),
            bias: bias,
            fstride: fstride,
            lookahead: lookahead
        )
    }

    func conv2dLayer(
        _ xBCHW: MLXArray,
        weight: MLXArray,
        bias: MLXArray?,
        fstride: Int,
        lookahead: Int
    ) throws -> MLXArray {
        conv2dLayer(
            xBCHW,
            weightOHWI: weight.transposed(0, 2, 3, 1),
            bias: bias,
            fstride: fstride,
            lookahead: lookahead
        )
    }

    func conv2dLayer(
        _ xBCHW: MLXArray,
        weightOHWI: MLXArray,
        bias: MLXArray?,
        fstride: Int,
        lookahead: Int
    ) -> MLXArray {
        Self.conv2dBCHW(
            xBCHW,
            weightOHWI: weightOHWI,
            bias: bias,
            fstride: fstride,
            lookahead: lookahead
        )
    }

    func convTranspose2dLayer(
        _ xBCHW: MLXArray,
        weightKey: String,
        fstride: Int,
        groups: Int
    ) throws -> MLXArray {
        if groups > 1, let denseWeight = convTransposeDenseWeights[weightKey] {
            return Self.convTranspose2dBCHWDense(
                xBCHW,
                denseWeight: denseWeight,
                fstride: fstride
            )
        }
        if groups > 1, let groupedWeights = convTransposeGroupWeights[weightKey], groupedWeights.count == groups {
            return convTranspose2dLayer(
                xBCHW,
                groupedWeights: groupedWeights,
                fstride: fstride
            )
        }
        return try convTranspose2dLayer(
            xBCHW,
            weight: w(weightKey),
            fstride: fstride,
            groups: groups
        )
    }

    func convTranspose2dLayer(
        _ xBCHW: MLXArray,
        weight: MLXArray,
        fstride: Int,
        groups: Int
    ) throws -> MLXArray {
        var x = xBCHW.transposed(0, 2, 3, 1)  // NHWC
        let kT = weight.shape[2]
        let kF = weight.shape[3]
        let padding = IntOrPair((kT - 1, kF / 2))
        let outputPadding = IntOrPair((0, kF / 2))

        if groups <= 1 {
            let w = weight.transposed(1, 2, 3, 0)
            x = MLX.convTransposed2d(
                x,
                w,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            return x.transposed(0, 3, 1, 2)
        }

        let inPerGroup = max(1, x.shape[3] / groups)
        let outPerGroup = weight.shape[1]
        var ys = [MLXArray]()
        ys.reserveCapacity(groups)

        for g in 0..<groups {
            let inStart = g * inPerGroup
            let inEnd = inStart + inPerGroup
            let xg = x[0..., 0..., 0..., inStart..<inEnd]

            let wg = weight[inStart..<inEnd, 0..., 0..., 0...]
            let wT = wg.transposed(1, 2, 3, 0)  // [out_pg, kT, kF, in_pg]
            let yg = MLX.convTransposed2d(
                xg,
                wT,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            ys.append(yg)
        }

        let y = MLX.concatenated(ys, axis: 3)
        _ = outPerGroup  // keep shape intent explicit
        return y.transposed(0, 3, 1, 2)
    }

    func convTranspose2dLayer(
        _ xBCHW: MLXArray,
        groupedWeights: [MLXArray],
        fstride: Int
    ) -> MLXArray {
        let groups = groupedWeights.count
        var x = xBCHW.transposed(0, 2, 3, 1)  // NHWC
        let kT = groupedWeights[0].shape[1]
        let kF = groupedWeights[0].shape[2]
        let padding = IntOrPair((kT - 1, kF / 2))
        let outputPadding = IntOrPair((0, kF / 2))
        let inPerGroup = max(1, x.shape[3] / groups)

        var ys = [MLXArray]()
        ys.reserveCapacity(groups)
        for (g, wT) in groupedWeights.enumerated() {
            let inStart = g * inPerGroup
            let inEnd = inStart + inPerGroup
            let xg = x[0..., 0..., 0..., inStart..<inEnd]
            let yg = MLX.convTransposed2d(
                xg,
                wT,
                stride: [1, fstride],
                padding: padding,
                outputPadding: outputPadding,
                groups: 1
            )
            ys.append(yg)
        }
        x = MLX.concatenated(ys, axis: 3)
        return x.transposed(0, 3, 1, 2)
    }

    func groupedConv2d(
        input: MLXArray,
        weight: MLXArray,
        strideW: Int,
        groups: Int
    ) throws -> MLXArray {
        MLX.conv2d(input, weight, stride: [1, strideW], padding: [0, 0], groups: groups)
    }

    func batchNorm(_ x: MLXArray, prefix: String) throws -> MLXArray {
        if let scale = bnScale[prefix], let bias = bnBias[prefix] {
            return x * scale + bias
        }

        let gamma = try w("\(prefix).weight")
        let beta = try w("\(prefix).bias")
        let mean = try w("\(prefix).running_mean")
        let variance = try w("\(prefix).running_var")
        let scale = (gamma / MLX.sqrt(variance + MLXArray(Float(1e-5))))
            .reshaped([1, gamma.shape[0], 1, 1])
        let shift = (beta - mean * (gamma / MLX.sqrt(variance + MLXArray(Float(1e-5)))))
            .reshaped([1, beta.shape[0], 1, 1])
        return x * scale + shift
    }

    func squeezedGRU(
        _ x: MLXArray,
        prefix: String,
        hiddenSize: Int,
        linearOut: Bool
    ) throws -> MLXArray {
        var y = relu(groupedLinear(x, weight: try w("\(prefix).linear_in.0.weight")))

        var layer = 0
        while weights["\(prefix).gru.weight_ih_l\(layer)"] != nil {
            y = try pytorchGRULayer(y, prefix: "\(prefix).gru", layer: layer, hiddenSize: hiddenSize)
            layer += 1
        }

        if linearOut, weights["\(prefix).linear_out.0.weight"] != nil {
            y = relu(groupedLinear(y, weight: try w("\(prefix).linear_out.0.weight")))
        }
        return y
    }

    func groupedLinear(_ x: MLXArray, weight: MLXArray) -> MLXArray {
        let groups = weight.shape[0]
        let ws = weight.shape[1]
        let hs = weight.shape[2]
        let b = x.shape[0]
        let t = x.shape[1]
        let xBTG1I = x.reshaped([b, t, groups, 1, ws])
        let w11GIH = weight
            .expandedDimensions(axis: 0)
            .expandedDimensions(axis: 0)
        let out = MLX.matmul(xBTG1I, w11GIH).squeezed(axis: 3)
        return out.reshaped([b, t, groups * hs])
    }

    func pytorchGRULayer(
        _ x: MLXArray,
        prefix: String,
        layer: Int,
        hiddenSize: Int
    ) throws -> MLXArray {
        let wihKey = "\(prefix).weight_ih_l\(layer)"
        let whhKey = "\(prefix).weight_hh_l\(layer)"
        let wihT: MLXArray
        if let cached = gruTransposedWeights[wihKey] {
            wihT = cached
        } else {
            wihT = try w(wihKey).transposed()
        }
        let whhT: MLXArray
        if let cached = gruTransposedWeights[whhKey] {
            whhT = cached
        } else {
            whhT = try w(whhKey).transposed()
        }
        let bih = try w("\(prefix).bias_ih_l\(layer)")
        let bhh = try w("\(prefix).bias_hh_l\(layer)")

        if performanceConfig.preferCompiledGraphs, x.shape[1] <= 256 {
            return compiledPytorchGRULayer(
                x,
                prefix: prefix,
                layer: layer,
                hiddenSize: hiddenSize,
                wihT: wihT,
                whhT: whhT,
                bih: bih,
                bhh: bhh
            )
        }

        let b = x.shape[0]
        let t = x.shape[1]
        let iSize = x.shape[2]
        let x2D = x.reshaped([b * t, iSize])
        let gxAll = (MLX.matmul(x2D, wihT) + bih).reshaped([b, t, 3 * hiddenSize])

        var h = MLXArray.zeros([b, hiddenSize], type: Float.self)
        var states = [MLXArray]()
        states.reserveCapacity(t)
        let one = MLXArray(Float(1.0)).asType(h.dtype)

        for i in 0..<t {
            let gx = gxAll[0..., i, 0...]
            let gh = MLX.addMM(bhh, h, whhT)

            let xr = gx[0..., 0..<hiddenSize]
            let xz = gx[0..., hiddenSize..<(2 * hiddenSize)]
            let xn = gx[0..., (2 * hiddenSize)...]
            let hr = gh[0..., 0..<hiddenSize]
            let hz = gh[0..., hiddenSize..<(2 * hiddenSize)]
            let hn = gh[0..., (2 * hiddenSize)...]

            let r = sigmoid(xr + hr)
            let z = sigmoid(xz + hz)
            let n = tanh(xn + r * hn)
            h = (one - z) * n + z * h
            states.append(h)
        }
        return MLX.stacked(states, axis: 1)
    }

    func compiledPytorchGRULayer(
        _ x: MLXArray,
        prefix: String,
        layer: Int,
        hiddenSize: Int,
        wihT: MLXArray,
        whhT: MLXArray,
        bih: MLXArray,
        bhh: MLXArray
    ) -> MLXArray {
        let b = x.shape[0]
        let t = x.shape[1]
        let iSize = x.shape[2]
        let cacheKey =
            "gru:\(prefix).\(layer):b\(b):t\(t):i\(iSize):h\(hiddenSize):\(x.dtype)"

        let compiled: CompiledArrayGraph
        if let cached = compiledGRULayerCache[cacheKey] {
            compiled = cached
        } else {
            compiled = compile(shapeless: false) { inputs in
                let xIn = inputs[0]
                let x2D = xIn.reshaped([b * t, iSize])
                let gxAll = (MLX.matmul(x2D, wihT) + bih).reshaped([b, t, 3 * hiddenSize])

                var h = MLXArray.zeros([b, hiddenSize], dtype: xIn.dtype)
                var states = [MLXArray]()
                states.reserveCapacity(t)
                let one = MLXArray(Float(1.0)).asType(xIn.dtype)

                for i in 0..<t {
                    let gx = gxAll[0..., i, 0...]
                    let gh = MLX.addMM(bhh, h, whhT)

                    let xr = gx[0..., 0..<hiddenSize]
                    let xz = gx[0..., hiddenSize..<(2 * hiddenSize)]
                    let xn = gx[0..., (2 * hiddenSize)...]
                    let hr = gh[0..., 0..<hiddenSize]
                    let hz = gh[0..., hiddenSize..<(2 * hiddenSize)]
                    let hn = gh[0..., (2 * hiddenSize)...]

                    let r = sigmoid(xr + hr)
                    let z = sigmoid(xz + hz)
                    let n = tanh(xn + r * hn)
                    h = (one - z) * n + z * h
                    states.append(h)
                }

                return [MLX.stacked(states, axis: 1)]
            }
            compiledGRULayerCache[cacheKey] = compiled
        }

        return compiled([x])[0]
    }

    func decodeErbCompiled(
        emb: MLXArray,
        e3: MLXArray,
        e2: MLXArray,
        e1: MLXArray,
        e0: MLXArray
    ) throws -> MLXArray? {
        var embDec = try squeezedGRU(
            emb,
            prefix: "erb_dec.emb_gru",
            hiddenSize: config.embHiddenDim,
            linearOut: true
        )

        let b = embDec.shape[0]
        let t = embDec.shape[1]
        let f8 = e3.shape[3]
        embDec = embDec.reshaped([b, t, f8, -1]).transposed(0, 3, 1, 2)

        guard let convt2Dense = try denseTransposeWeight("erb_dec.convt2.0.weight"),
              let convt1Dense = try denseTransposeWeight("erb_dec.convt1.0.weight")
        else {
            return nil
        }

        let conv3pW = try convWeightOHWI("erb_dec.conv3p.0.weight")
        let convt3W0 = try convWeightOHWI("erb_dec.convt3.0.weight")
        let convt3W1 = try convWeightOHWI("erb_dec.convt3.1.weight")
        let conv2pW = try convWeightOHWI("erb_dec.conv2p.0.weight")
        let convt2W1 = try convWeightOHWI("erb_dec.convt2.1.weight")
        let conv1pW = try convWeightOHWI("erb_dec.conv1p.0.weight")
        let convt1W1 = try convWeightOHWI("erb_dec.convt1.1.weight")
        let conv0pW = try convWeightOHWI("erb_dec.conv0p.0.weight")
        let conv0OutW = try convWeightOHWI("erb_dec.conv0_out.0.weight")

        let conv3pBN = try batchNormAffine(prefix: "erb_dec.conv3p.1")
        let convt3BN = try batchNormAffine(prefix: "erb_dec.convt3.2")
        let conv2pBN = try batchNormAffine(prefix: "erb_dec.conv2p.1")
        let convt2BN = try batchNormAffine(prefix: "erb_dec.convt2.2")
        let conv1pBN = try batchNormAffine(prefix: "erb_dec.conv1p.1")
        let convt1BN = try batchNormAffine(prefix: "erb_dec.convt1.2")
        let conv0pBN = try batchNormAffine(prefix: "erb_dec.conv0p.1")
        let conv0OutBN = try batchNormAffine(prefix: "erb_dec.conv0_out.1")

        let cacheKey =
            "erbdec:b\(b):t\(t):f0\(e0.shape[3]):f1\(e1.shape[3]):f2\(e2.shape[3]):f3\(e3.shape[3]):\(embDec.dtype)"
        let compiled: CompiledArrayGraph
        if let cached = compiledErbDecoderCache[cacheKey] {
            compiled = cached
        } else {
            compiled = compile(shapeless: false) { inputs in
                let embDecIn = inputs[0]
                let e3In = inputs[1]
                let e2In = inputs[2]
                let e1In = inputs[3]
                let e0In = inputs[4]

                let p3 = relu(Self.conv2dBCHW(e3In, weightOHWI: conv3pW, bias: nil, fstride: 1, lookahead: 0) * conv3pBN.0 + conv3pBN.1)
                var d3 = p3 + embDecIn
                d3 = Self.conv2dBCHW(d3, weightOHWI: convt3W0, bias: nil, fstride: 1, lookahead: 0)
                d3 = Self.conv2dBCHW(d3, weightOHWI: convt3W1, bias: nil, fstride: 1, lookahead: 0)
                d3 = relu(d3 * convt3BN.0 + convt3BN.1)

                let p2 = relu(Self.conv2dBCHW(e2In, weightOHWI: conv2pW, bias: nil, fstride: 1, lookahead: 0) * conv2pBN.0 + conv2pBN.1)
                var d2 = p2 + d3
                d2 = Self.convTranspose2dBCHWDense(d2, denseWeight: convt2Dense, fstride: 2)
                d2 = Self.conv2dBCHW(d2, weightOHWI: convt2W1, bias: nil, fstride: 1, lookahead: 0)
                d2 = relu(d2 * convt2BN.0 + convt2BN.1)

                let p1 = relu(Self.conv2dBCHW(e1In, weightOHWI: conv1pW, bias: nil, fstride: 1, lookahead: 0) * conv1pBN.0 + conv1pBN.1)
                var d1 = p1 + d2
                d1 = Self.convTranspose2dBCHWDense(d1, denseWeight: convt1Dense, fstride: 2)
                d1 = Self.conv2dBCHW(d1, weightOHWI: convt1W1, bias: nil, fstride: 1, lookahead: 0)
                d1 = relu(d1 * convt1BN.0 + convt1BN.1)

                let p0 = relu(Self.conv2dBCHW(e0In, weightOHWI: conv0pW, bias: nil, fstride: 1, lookahead: 0) * conv0pBN.0 + conv0pBN.1)
                let d0 = p0 + d1
                var out = Self.conv2dBCHW(d0, weightOHWI: conv0OutW, bias: nil, fstride: 1, lookahead: 0)
                out = out * conv0OutBN.0 + conv0OutBN.1
                return [sigmoid(out)]
            }
            compiledErbDecoderCache[cacheKey] = compiled
        }

        return compiled([embDec, e3, e2, e1, e0])[0]
    }

    func linear(_ x: MLXArray, weight: MLXArray, bias: MLXArray) -> MLXArray {
        let b = x.shape[0]
        let t = x.shape[1]
        let x2 = x.reshaped([b * t, x.shape[2]])
        var y = MLX.matmul(x2, weight.transposed())
        y = y + bias
        return y.reshaped([b, t, weight.shape[0]])
    }

}
