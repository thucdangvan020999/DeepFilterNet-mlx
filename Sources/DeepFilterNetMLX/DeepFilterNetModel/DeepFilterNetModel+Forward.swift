import Foundation
import HuggingFace
import MLX
import MLXNN

extension DeepFilterNetModel {
    // MARK: - Network Forward

    func forward(
        spec: MLXArray,
        featErb: MLXArray,
        featSpec5D: MLXArray
    ) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let featSpec = featSpec5D
            .squeezed(axis: 1)
            .transposed(0, 3, 1, 2)

        let featErbShift = applyLookahead(feature: featErb, lookahead: config.convLookahead)
        let featSpecShift = applyLookahead(feature: featSpec, lookahead: config.convLookahead)

        let (e0, e1, e2, e3, emb, c0, lsnr) = try encode(featErb: featErbShift, featSpec: featSpecShift)

        let mask = try decodeErb(emb: emb, e3: e3, e2: e2, e1: e1, e0: e0)
        let specMasked = applyMask(spec: spec, mask: mask)

        let dfCoefs = try decodeDf(emb: emb, c0: c0)
        let b = dfCoefs.shape[0]
        let t = dfCoefs.shape[1]
        let dfCoefs5 = dfCoefs
            .reshaped([b, t, config.nbDf, config.dfOrder, 2])
            .transposed(0, 3, 1, 2, 4)

        let specEnhanced: MLXArray
        if config.encConcat {
            specEnhanced = deepFilter(spec: specMasked, coefs: dfCoefs5)
        } else {
            let specDf = deepFilter(spec: spec, coefs: dfCoefs5)
            let low = specDf[0..., 0..., 0..., 0..<config.nbDf, 0...]
            let high = specMasked[0..., 0..., 0..., config.nbDf..., 0...]
            specEnhanced = MLX.concatenated([low, high], axis: 3)
        }

        return (specEnhanced, mask, lsnr, dfCoefs5)
    }

    func forwardV1(
        spec: MLXArray,
        featErb: MLXArray,
        featSpec5D: MLXArray
    ) throws -> (MLXArray, MLXArray, MLXArray, MLXArray) {
        let featSpec = featSpec5D
            .squeezed(axis: 1)
            .transposed(0, 3, 1, 2)

        let (e0, e1, e2, e3, emb, c0, lsnr) = try encodeV1(featErb: featErb, featSpec: featSpec)
        var mask = try decodeErbV1(emb: emb, e3: e3, e2: e2, e1: e1, e0: e0)
        mask = alignTimeAxis(mask, target: spec.shape[2], fillValue: 1.0, axis: 2)

        let specMasked = applyMask(spec: spec, mask: mask)
        var (dfCoefsBTOF2, dfAlpha) = try decodeDfV1(emb: emb, c0: c0)
        dfCoefsBTOF2 = alignTimeAxis(dfCoefsBTOF2, target: spec.shape[2], fillValue: 0.0, axis: 1)
        dfAlpha = alignTimeAxis(dfAlpha, target: spec.shape[2], fillValue: 0.0, axis: 1)
        let dfCoefs5 = dfCoefsBTOF2.transposed(0, 2, 1, 3, 4)

        let specEnhanced = deepFilter(spec: specMasked, coefs: dfCoefs5, alpha: dfAlpha)
        return (specEnhanced, mask, lsnr, dfCoefs5)
    }

    func encodeV1(featErb: MLXArray, featSpec: MLXArray)
        throws -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
    {
        let e0 = try applyV1ConvKxF(
            featErb,
            prefix: "enc.erb_conv0",
            fstride: 1,
            lookahead: config.convLookahead > 0 ? 1 : 0,
            activation: .relu
        )
        let e1 = try applyV1ConvKxF(
            e0,
            prefix: "enc.erb_conv1",
            fstride: 2,
            lookahead: config.convLookahead > 1 ? 1 : 0,
            activation: .relu
        )
        let e2 = try applyV1ConvKxF(
            e1,
            prefix: "enc.erb_conv2",
            fstride: 2,
            lookahead: config.convLookahead > 2 ? 1 : 0,
            activation: .relu
        )
        let e3 = try applyV1ConvKxF(
            e2,
            prefix: "enc.erb_conv3",
            fstride: 1,
            lookahead: 0,
            activation: .relu
        )

        let c0 = try applyV1ConvKxF(
            featSpec,
            prefix: "enc.clc_conv0",
            fstride: 1,
            lookahead: config.convLookahead,
            activation: .relu
        )
        let c1 = try applyV1ConvKxF(
            c0,
            prefix: "enc.clc_conv1",
            fstride: 2,
            lookahead: 0,
            activation: .relu
        )

        let t = c1.shape[2]
        let b = c1.shape[0]
        var cemb = c1.transposed(2, 0, 1, 3).reshaped([t, b, -1])
        cemb = relu(try groupedLinearV1(cemb, prefix: "enc.clc_fc_emb.layers"))

        var emb = e3.transposed(2, 0, 1, 3).reshaped([t, b, -1])
        emb = emb + cemb
        emb = try groupedGRUV1(
            emb,
            prefix: "enc.emb_gru.grus",
            numLayers: config.embNumLayers,
            addOutputs: true,
            shuffleBetweenLayers: config.groupShuffle
        )
        let embBT = emb.transposed(1, 0, 2)

        let lsnrScale = MLXArray(Float(config.lsnrMax - config.lsnrMin)).asType(emb.dtype)
        let lsnrOffset = MLXArray(Float(config.lsnrMin)).asType(emb.dtype)
        let lsnr = sigmoid(linear(
            embBT,
            weight: try w("enc.lsnr_fc.0.weight"),
            bias: try w("enc.lsnr_fc.0.bias")
        )) * lsnrScale + lsnrOffset

        return (e0, e1, e2, e3, embBT, c0, lsnr)
    }

    func decodeErbV1(
        emb: MLXArray,
        e3: MLXArray,
        e2: MLXArray,
        e1: MLXArray,
        e0: MLXArray
    ) throws -> MLXArray {
        let b = emb.shape[0]
        let t = emb.shape[1]
        let f8 = e3.shape[3]

        var embProj = relu(try groupedLinearV1(emb, prefix: "erb_dec.fc_emb.0.layers"))
        embProj = embProj.reshaped([b, t, -1, f8]).transposed(0, 2, 1, 3)

        let p3 = try applyV1ConvKxF(e3, prefix: "erb_dec.conv3p", fstride: 1, lookahead: 0, activation: .relu)
        var d3 = alignAndAdd(p3, embProj)
        d3 = try applyV1ConvKxF(d3, prefix: "erb_dec.convt3", fstride: 1, lookahead: 0, activation: .relu)

        let p2 = try applyV1ConvKxF(e2, prefix: "erb_dec.conv2p", fstride: 1, lookahead: 0, activation: .relu)
        var d2 = alignAndAdd(p2, d3)
        d2 = try applyV1ConvKxF(d2, prefix: "erb_dec.convt2", fstride: 2, lookahead: 0, activation: .relu)

        let p1 = try applyV1ConvKxF(e1, prefix: "erb_dec.conv1p", fstride: 1, lookahead: 0, activation: .relu)
        var d1 = alignAndAdd(p1, d2)
        d1 = try applyV1ConvKxF(d1, prefix: "erb_dec.convt1", fstride: 2, lookahead: 0, activation: .relu)

        let p0 = try applyV1ConvKxF(e0, prefix: "erb_dec.conv0p", fstride: 1, lookahead: 0, activation: .relu)
        let d0 = alignAndAdd(p0, d1)

        return try applyV1ConvKxF(
            d0,
            prefix: "erb_dec.conv0_out",
            fstride: 1,
            lookahead: 0,
            activation: .sigmoid,
            applyBatchNorm: false,
            allowBias: true
        )
    }

    func decodeDfV1(emb: MLXArray, c0: MLXArray) throws -> (MLXArray, MLXArray) {
        let cTBI = try groupedGRUV1(
            emb.transposed(1, 0, 2),
            prefix: "clc_dec.clc_gru.grus",
            numLayers: config.dfNumLayers,
            addOutputs: true,
            shuffleBetweenLayers: config.groupShuffle
        )
        let c = cTBI.transposed(1, 0, 2)

        var c0p = try applyV1ConvKxF(
            c0,
            prefix: "clc_dec.clc_convp",
            fstride: 1,
            lookahead: 0,
            activation: .relu
        )
        c0p = c0p.transposed(0, 2, 1, 3)  // [B, T, O*2, F]

        let b = c.shape[0]
        let t = c.shape[1]
        let alpha = sigmoid(linear(
            c,
            weight: try w("clc_dec.clc_fc_a.0.weight"),
            bias: try w("clc_dec.clc_fc_a.0.bias")
        ))

        var coefs = tanh(linear(
            c,
            weight: try w("clc_dec.clc_fc_out.0.weight"),
            bias: try w("clc_dec.clc_fc_out.0.bias")
        ))
        coefs = coefs.reshaped([b, t, config.dfOrder * 2, config.nbDf]) + c0p
        coefs = coefs.reshaped([b, t, config.dfOrder, 2, config.nbDf]).transposed(0, 1, 2, 4, 3)
        return (coefs, alpha)
    }

    func encode(featErb: MLXArray, featSpec: MLXArray)
        throws -> (MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray, MLXArray)
    {
        let e0 = try applyEncoderConv(featErb, prefix: "enc.erb_conv0", main: 1, pointwise: nil, bn: 2, fstride: 1)
        let e1 = try applyEncoderConv(e0, prefix: "enc.erb_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)
        let e2 = try applyEncoderConv(e1, prefix: "enc.erb_conv2", main: 0, pointwise: 1, bn: 2, fstride: 2)
        let e3 = try applyEncoderConv(e2, prefix: "enc.erb_conv3", main: 0, pointwise: 1, bn: 2, fstride: 1)

        let c0 = try applyEncoderConv(featSpec, prefix: "enc.df_conv0", main: 1, pointwise: 2, bn: 3, fstride: 1)
        let c1 = try applyEncoderConv(c0, prefix: "enc.df_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2)

        let b = c1.shape[0]
        let t = c1.shape[2]
        var cemb = c1.transposed(0, 2, 3, 1).reshaped([b, t, -1])
        cemb = relu(groupedLinear(cemb, weight: try w("enc.df_fc_emb.0.weight")))

        var emb = e3.transposed(0, 2, 3, 1).reshaped([b, t, -1])
        emb = config.encConcat ? MLX.concatenated([emb, cemb], axis: -1) : (emb + cemb)

        emb = try squeezedGRU(
            emb,
            prefix: "enc.emb_gru",
            hiddenSize: config.embHiddenDim,
            linearOut: true
        )

        let lsnrScale = MLXArray(Float(config.lsnrMax - config.lsnrMin)).asType(emb.dtype)
        let lsnrOffset = MLXArray(Float(config.lsnrMin)).asType(emb.dtype)
        let lsnr = sigmoid(linear(
            emb,
            weight: try w("enc.lsnr_fc.0.weight"),
            bias: try w("enc.lsnr_fc.0.bias")
        )) * lsnrScale + lsnrOffset

        return (e0, e1, e2, e3, emb, c0, lsnr)
    }

    func decodeErb(
        emb: MLXArray,
        e3: MLXArray,
        e2: MLXArray,
        e1: MLXArray,
        e0: MLXArray
    ) throws -> MLXArray {
        if performanceConfig.preferCompiledGraphs,
           emb.shape[1] <= 256,
           let fused = try decodeErbCompiled(
               emb: emb,
               e3: e3,
               e2: e2,
               e1: e1,
               e0: e0
           )
        {
            return fused
        }

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

        var d3 = relu(try applyPathwayConv(e3, prefix: "erb_dec.conv3p")) + embDec
        // Matches MLX/PyTorch DF decoder: convt3 is a regular conv block, not transposed.
        d3 = relu(try applyRegularBlock(d3, prefix: "erb_dec.convt3"))
        var d2 = relu(try applyPathwayConv(e2, prefix: "erb_dec.conv2p")) + d3
        d2 = relu(try applyTransposeBlock(d2, prefix: "erb_dec.convt2", fstride: 2))
        var d1 = relu(try applyPathwayConv(e1, prefix: "erb_dec.conv1p")) + d2
        d1 = relu(try applyTransposeBlock(d1, prefix: "erb_dec.convt1", fstride: 2))
        let d0 = relu(try applyPathwayConv(e0, prefix: "erb_dec.conv0p")) + d1
        let out = try applyOutputConv(d0, prefix: "erb_dec.conv0_out")
        return sigmoid(out)
    }

    func decodeDf(emb: MLXArray, c0: MLXArray) throws -> MLXArray {
        var c = try squeezedGRU(
            emb,
            prefix: "df_dec.df_gru",
            hiddenSize: config.dfHiddenDim,
            linearOut: false
        )

        if weights["df_dec.df_skip.weight"] != nil {
            c = c + groupedLinear(emb, weight: try w("df_dec.df_skip.weight"))
        }

        var c0p = try conv2dLayer(
            c0,
            weightKey: "df_dec.df_convp.1.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        c0p = try conv2dLayer(
            c0p,
            weightKey: "df_dec.df_convp.2.weight",
            bias: nil,
            fstride: 1,
            lookahead: 0
        )
        c0p = relu(try batchNorm(c0p, prefix: "df_dec.df_convp.3"))
        c0p = c0p.transposed(0, 2, 3, 1)

        let b = c.shape[0]
        let t = c.shape[1]
        let dfOut = tanh(groupedLinear(c, weight: try w("df_dec.df_out.0.weight")))
            .reshaped([b, t, config.nbDf, config.dfOrder * 2])

        return dfOut + c0p
    }

    func applyMask(spec: MLXArray, mask: MLXArray) -> MLXArray {
        if performanceConfig.enableMetalFusedErbInvMaskApply,
           spec.dtype == mask.dtype,
           (spec.dtype == .float32 || spec.dtype == .float16),
           let fused = DeepFilterNetKernelFusion.shared.applyMaskErbInv(
               spec: spec,
               mask: mask,
               erbInvFB: spec.dtype == .float16 ? erbInvFBF16 : erbInvFBF32,
               threadGroupSize: performanceConfig.kernelThreadGroupSize
           )
        {
            return fused
        }
        let b = mask.shape[0]
        let t = mask.shape[2]
        let e = mask.shape[3]
        let flat = mask.reshaped([b * t, e])
        let erbInv: MLXArray
        if spec.dtype == .float16 {
            erbInv = erbInvFBF16
        } else if spec.dtype == .float32 {
            erbInv = erbInvFBF32
        } else {
            erbInv = erbInvFB.asType(spec.dtype)
        }
        let gains = MLX.matmul(flat, erbInv).reshaped([b, 1, t, config.freqBins, 1])
        if performanceConfig.enableMetalFusedMaskMultiply,
           let fused = DeepFilterNetKernelFusion.shared.applyMaskMultiply(
               spec: spec,
               gains: gains,
               threadGroupSize: performanceConfig.kernelThreadGroupSize,
               ensureContiguous: performanceConfig.ensureKernelContiguousInputs
           )
        {
            return fused
        }
        return spec * gains
    }

    func deepFilter(spec: MLXArray, coefs: MLXArray, alpha: MLXArray? = nil) -> MLXArray {
        let t = spec.shape[2]
        let padLeft = config.dfOrder - 1 - config.dfLookahead

        let specLow = spec[0..., 0, 0..., 0..<config.nbDf, 0...]
        let lowRaw: MLXArray
        if performanceConfig.enableMetalFusedOfflineDeepFilter,
           let fused = DeepFilterNetKernelFusion.shared.deepFilterOffline(
               specLow: specLow,
               coefs: coefs,
               padLeft: padLeft,
               threadGroupSize: performanceConfig.kernelThreadGroupSize,
               ensureContiguous: performanceConfig.ensureKernelContiguousInputs
           )
        {
            lowRaw = fused
        } else {
            let padRight = config.dfLookahead
            let padded = MLX.padded(
                specLow,
                widths: [
                    .init(0),
                    .init((padLeft, padRight)),
                    .init(0),
                    .init(0),
                ],
                mode: .constant
            )

            let b = spec.shape[0]
            var outR = MLXArray.zeros([b, t, config.nbDf], dtype: spec.dtype)
            var outI = MLXArray.zeros([b, t, config.nbDf], dtype: spec.dtype)
            for k in 0..<config.dfOrder {
                let window = padded[0..., k..<(k + t), 0..., 0...]  // [B, T, F_df, 2]
                let coef = coefs[0..., k, 0..., 0..., 0...]  // [B, T, F_df, 2]
                let sr = window[0..., 0..., 0..., 0]
                let si = window[0..., 0..., 0..., 1]
                let cr = coef[0..., 0..., 0..., 0]
                let ci = coef[0..., 0..., 0..., 1]

                outR = outR + (sr * cr - si * ci)
                outI = outI + (sr * ci + si * cr)
            }
            lowRaw = MLX.stacked([outR, outI], axis: -1)
        }

        var low = lowRaw.expandedDimensions(axis: 1)
        if let alpha {
            let b = spec.shape[0]
            let a = alpha.reshaped([b, 1, t, 1, 1])
            let origLow = spec[0..., 0..., 0..., 0..<config.nbDf, 0...]
            let one = MLXArray(Float(1.0)).asType(a.dtype)
            low = low * a + origLow * (one - a)
        }
        let high = spec[0..., 0..., 0..., config.nbDf..., 0...]
        return MLX.concatenated([low, high], axis: 3)
    }

    enum V1Activation {
        case none
        case relu
        case sigmoid
    }

    func applyV1ConvKxF(
        _ x: MLXArray,
        prefix: String,
        fstride: Int,
        lookahead: Int,
        activation: V1Activation,
        applyBatchNorm: Bool = true,
        allowBias: Bool = false
    ) throws -> MLXArray {
        let mainKey: String
        let transposed: Bool
        if weights["\(prefix).sconvt.weight"] != nil {
            mainKey = "\(prefix).sconvt.weight"
            transposed = true
        } else {
            mainKey = "\(prefix).sconv.weight"
            transposed = false
        }

        let mainWeight = try w(mainKey)
        var y: MLXArray
        if transposed {
            let groups = max(1, mainWeight.shape[0] / max(1, mainWeight.shape[1]))
            y = try convTranspose2dLayer(
                x,
                weight: mainWeight,
                fstride: fstride,
                groups: groups
            )
        } else {
            let bias = allowBias ? weights["\(prefix).sconv.bias"] : nil
            y = try conv2dLayer(
                x,
                weight: mainWeight,
                bias: bias,
                fstride: fstride,
                lookahead: lookahead
            )
        }

        if weights["\(prefix).1x1conv.weight"] != nil {
            y = try conv2dLayer(
                y,
                weightKey: "\(prefix).1x1conv.weight",
                bias: nil,
                fstride: 1,
                lookahead: 0
            )
        }

        if applyBatchNorm, weights["\(prefix).norm.running_mean"] != nil {
            y = try batchNorm(y, prefix: "\(prefix).norm")
        }

        switch activation {
        case .none:
            return y
        case .relu:
            return relu(y)
        case .sigmoid:
            return sigmoid(y)
        }
    }

    func groupedLinearV1(
        _ x: MLXArray,
        prefix: String
    ) throws -> MLXArray {
        if let pack = v1GroupedLinearPacks[prefix] {
            let b = x.shape[0]
            let t = x.shape[1]
            var y = MLX.einsum(
                "btgi,gio->btgo",
                x.reshaped([b, t, pack.groups, pack.inputPerGroup]),
                pack.weightGIO
            ) + pack.biasGO.reshaped([1, 1, pack.groups, pack.outputPerGroup])
            y = y.reshaped([b, t, pack.groups * pack.outputPerGroup])
            if config.groupShuffle, pack.groups > 1 {
                let hiddenPerGroup = y.shape[2] / pack.groups
                y = y
                    .reshaped([y.shape[0], y.shape[1], hiddenPerGroup, pack.groups])
                    .transposed(0, 1, 3, 2)
                    .reshaped([y.shape[0], y.shape[1], -1])
            }
            return y
        }

        let groups = max(1, config.linearGroups)
        var ys = [MLXArray]()
        ys.reserveCapacity(groups)
        for g in 0..<groups {
            let wg = try w("\(prefix).\(g).weight")
            let bg = try w("\(prefix).\(g).bias")
            let inPerGroup = wg.shape[1]
            let start = g * inPerGroup
            let stop = start + inPerGroup
            let xg = x[0..., 0..., start..<stop]
            let x2 = xg.reshaped([-1, inPerGroup])
            let y2 = MLX.addMM(bg, x2, wg.transposed())
            ys.append(y2.reshaped([x.shape[0], x.shape[1], wg.shape[0]]))
        }
        var y = MLX.concatenated(ys, axis: 2)
        if config.groupShuffle, groups > 1 {
            let hiddenPerGroup = y.shape[2] / groups
            y = y
                .reshaped([y.shape[0], y.shape[1], hiddenPerGroup, groups])
                .transposed(0, 1, 3, 2)
                .reshaped([y.shape[0], y.shape[1], -1])
        }
        return y
    }

    func groupedGRUV1(
        _ xTBI: MLXArray,
        prefix: String,
        numLayers: Int,
        addOutputs: Bool,
        shuffleBetweenLayers: Bool
    ) throws -> MLXArray {
        if let pack = v1GroupedGRUPacks[prefix], pack.layers.count >= numLayers {
            return groupedGRUV1Packed(
                xTBI,
                pack: pack,
                numLayers: numLayers,
                addOutputs: addOutputs,
                shuffleBetweenLayers: shuffleBetweenLayers
            )
        }

        let groups = max(1, config.gruGroups)
        var cur = xTBI
        var out = xTBI

        for layer in 0..<numLayers {
            let base = "\(prefix).\(layer).layers"
            let w0 = try w("\(base).0.weight_ih_l0")
            let inPerGroup = w0.shape[1]
            let hiddenPerGroup = w0.shape[0] / 3
            var ys = [MLXArray]()
            ys.reserveCapacity(groups)
            for g in 0..<groups {
                let start = g * inPerGroup
                let stop = start + inPerGroup
                let xg = cur[0..., 0..., start..<stop]
                let yg = try gruCellSequenceV1(
                    xg,
                    weightIH: w("\(base).\(g).weight_ih_l0"),
                    weightHH: w("\(base).\(g).weight_hh_l0"),
                    biasIH: w("\(base).\(g).bias_ih_l0"),
                    biasHH: w("\(base).\(g).bias_hh_l0"),
                    hiddenSize: hiddenPerGroup
                )
                ys.append(yg)
            }

            var layerOut = MLX.concatenated(ys, axis: 2)
            if shuffleBetweenLayers && layer < numLayers - 1 && groups > 1 {
                let hidden = layerOut.shape[2] / groups
                layerOut = layerOut
                    .reshaped([layerOut.shape[0], layerOut.shape[1], hidden, groups])
                    .transposed(0, 1, 3, 2)
                    .reshaped([layerOut.shape[0], layerOut.shape[1], -1])
            }

            if addOutputs {
                out = (layer == 0) ? layerOut : (out + layerOut)
            } else {
                out = layerOut
            }
            cur = layerOut
        }

        return out
    }

    func groupedGRUV1Packed(
        _ xTBI: MLXArray,
        pack: V1GroupedGRUPack,
        numLayers: Int,
        addOutputs: Bool,
        shuffleBetweenLayers: Bool
    ) -> MLXArray {
        var cur = xTBI
        var out = xTBI
        let groups = pack.groups

        for layer in 0..<numLayers {
            let p = pack.layers[layer]
            let t = cur.shape[0]
            let b = cur.shape[1]
            let h = p.hiddenPerGroup
            let x4 = cur.reshaped([t, b, groups, p.inputPerGroup])

            var state = MLXArray.zeros([b, groups, h], dtype: cur.dtype)
            var ys = [MLXArray]()
            ys.reserveCapacity(t)
            for ti in 0..<t {
                let xt = x4[ti, 0..., 0..., 0...]  // [B, G, I]
                let gx = MLX.einsum("bgi,gio->bgo", xt, p.weightIHGI3H) + p.biasIHG3H.reshaped([1, groups, 3 * h])
                let gh = MLX.einsum("bgh,gho->bgo", state, p.weightHHGH3H) + p.biasHHG3H.reshaped([1, groups, 3 * h])

                let xr = gx[0..., 0..., 0..<h]
                let xz = gx[0..., 0..., h..<(2 * h)]
                let xn = gx[0..., 0..., (2 * h)...]
                let hr = gh[0..., 0..., 0..<h]
                let hz = gh[0..., 0..., h..<(2 * h)]
                let hn = gh[0..., 0..., (2 * h)...]

                let r = sigmoid(xr + hr)
                let z = sigmoid(xz + hz)
                let n = tanh(xn + r * hn)
                state = n + z * (state - n)
                ys.append(state)
            }

            var layerOut = MLX.stacked(ys, axis: 0).reshaped([t, b, groups * h])
            if shuffleBetweenLayers && layer < numLayers - 1 && groups > 1 {
                let hidden = layerOut.shape[2] / groups
                layerOut = layerOut
                    .reshaped([layerOut.shape[0], layerOut.shape[1], hidden, groups])
                    .transposed(0, 1, 3, 2)
                    .reshaped([layerOut.shape[0], layerOut.shape[1], -1])
            }

            if addOutputs {
                out = (layer == 0) ? layerOut : (out + layerOut)
            } else {
                out = layerOut
            }
            cur = layerOut
        }

        return out
    }

    func gruCellSequenceV1(
        _ xTBI: MLXArray,
        weightIH: @autoclosure () throws -> MLXArray,
        weightHH: @autoclosure () throws -> MLXArray,
        biasIH: @autoclosure () throws -> MLXArray,
        biasHH: @autoclosure () throws -> MLXArray,
        hiddenSize: Int
    ) throws -> MLXArray {
        let wihT = try weightIH().transposed()
        let whhT = try weightHH().transposed()
        let bih = try biasIH()
        let bhh = try biasHH()

        var h = MLXArray.zeros([xTBI.shape[1], hiddenSize], type: Float.self)
        var states = [MLXArray]()
        states.reserveCapacity(xTBI.shape[0])

        for i in 0..<xTBI.shape[0] {
            let xt = xTBI[i, 0..., 0...]
            let gx = MLX.addMM(bih, xt, wihT)
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
            h = n + z * (h - n)
            states.append(h)
        }
        return MLX.stacked(states, axis: 0)
    }

    func alignAndAdd(_ a: MLXArray, _ b: MLXArray) -> MLXArray {
        let t = min(a.shape[2], b.shape[2])
        let f = min(a.shape[3], b.shape[3])
        return a[0..., 0..., 0..<t, 0..<f] + b[0..., 0..., 0..<t, 0..<f]
    }

    func alignTimeAxis(
        _ x: MLXArray,
        target: Int,
        fillValue: Float,
        axis: Int
    ) -> MLXArray {
        let t = x.shape[axis]
        if t == target { return x }
        if t > target {
            switch axis {
            case 1:
                if x.ndim == 3 {
                    return x[0..., 0..<target, 0...]
                }
                return x[0..., 0..<target, 0..., 0..., 0...]
            case 2:
                if x.ndim == 4 {
                    return x[0..., 0..., 0..<target, 0...]
                }
                return x[0..., 0..., 0..<target, 0..., 0...]
            default:
                return x
            }
        }
        switch axis {
        case 1:
            if x.ndim == 3 {
                let pad = MLX.full([x.shape[0], target - t, x.shape[2]], values: fillValue)
                return MLX.concatenated([x, pad], axis: 1)
            }
            let pad = MLX.full([x.shape[0], target - t, x.shape[2], x.shape[3], x.shape[4]], values: fillValue)
            return MLX.concatenated([x, pad], axis: 1)
        case 2:
            if x.ndim == 4 {
                let pad = MLX.full([x.shape[0], x.shape[1], target - t, x.shape[3]], values: fillValue)
                return MLX.concatenated([x, pad], axis: 2)
            }
            let pad = MLX.full([x.shape[0], x.shape[1], target - t, x.shape[3], x.shape[4]], values: fillValue)
            return MLX.concatenated([x, pad], axis: 2)
        default:
            return x
        }
    }
}
