import Foundation
import HuggingFace
import MLX
import MLXNN

extension DeepFilterNetModel {
    // MARK: - Streaming

    public final class DeepFilterNetStreamer {
        final class StreamGRULayer {
            let wihT: MLXArray
            let whhT: MLXArray
            let bih: MLXArray
            let bhh: MLXArray
            var compiledStepCache: [String: (@Sendable ([MLXArray]) -> [MLXArray])] = [:]

            init(wihT: MLXArray, whhT: MLXArray, bih: MLXArray, bhh: MLXArray) {
                self.wihT = wihT
                self.whhT = whhT
                self.bih = bih
                self.bhh = bhh
            }
        }

        struct StreamGRU {
            let linearInWeight: MLXArray
            let layers: [StreamGRULayer]
            let linearOutWeight: MLXArray?
        }

        struct StreamConvLayer {
            let mainWeightOHWI: MLXArray
            let pointwiseWeightOHWI: MLXArray?
            let bnScale: MLXArray
            let bnBias: MLXArray
            let fstride: Int
        }

        let model: DeepFilterNetModel
        public let config: DeepFilterNetStreamingConfig

        let fftSize: Int
        let hopSize: Int
        let freqBins: Int
        let nbDf: Int
        let nbErb: Int
        let dfOrder: Int
        let dfLookahead: Int
        let convLookahead: Int

        let alphaArray: MLXArray
        let oneMinusAlphaArray: MLXArray
        let fftScaleArray: MLXArray
        let vorbisWindow: MLXArray
        let wnormArray: MLXArray
        let inferenceDType: DType
        let epsEnergy = MLXArray(Float(1e-10))
        let epsNorm = MLXArray(Float(1e-12))
        let tenArray = MLXArray(Float(10.0))
        let fortyArray = MLXArray(Float(40.0))
        let zeroSpecFrame: MLXArray
        let zeroSpecLowFrame: MLXArray
        let zeroSpecLowFrameInference: MLXArray
        let zeroMaskFrame: MLXArray
        let zeroMaskFrameInference: MLXArray
        let zeroEncErbFrame: MLXArray
        let zeroEncDfFrame: MLXArray
        let zeroDfConvpFrame: MLXArray
        let specRingCapacity: Int
        let dfConvpKernelSizeT: Int
        let dfSpecLeft: Int
        let analysisMemCount: Int
        let synthMemCount: Int
        let erbFBFrame: MLXArray?
        let lsnrWeight: MLXArray
        let lsnrBias: MLXArray
        let lsnrScale: MLXArray
        let lsnrOffset: MLXArray
        let encDfFcEmbWeight: MLXArray
        let dfDecSkipWeight: MLXArray?
        let dfDecOutWeight: MLXArray
        let encEmbGRU: StreamGRU
        let erbDecEmbGRU: StreamGRU
        let dfDecGRU: StreamGRU
        let encErbConv0: StreamConvLayer
        let encErbConv1: StreamConvLayer
        let encErbConv2: StreamConvLayer
        let encErbConv3: StreamConvLayer
        let encDfConv0: StreamConvLayer
        let encDfConv1: StreamConvLayer

        var pendingSamples = MLXArray.zeros([0], type: Float.self)
        var analysisMem: MLXArray
        var synthMem: MLXArray
        var erbState: MLXArray
        var dfState: MLXArray

        var rings: DeepFilterNetStreamingRings
        var recurrentState = DeepFilterNetStreamRecurrentState()

        var delayDropped = 0
        var hopsSinceMaterialize = 0
        let enableProfiling: Bool
        var profHopCount = 0
        var profAnalysisSeconds = 0.0
        var profFeaturesSeconds = 0.0
        var profInferSeconds = 0.0
        var profInferEncodeSeconds = 0.0
        var profInferEmbSeconds = 0.0
        var profInferErbSeconds = 0.0
        var profInferDfSeconds = 0.0
        var profSynthesisSeconds = 0.0
        var profMaterializeSeconds = 0.0
        let profilingForceEvalPerStage: Bool
        let compiledAnalysisFeatureStep: (@Sendable ([MLXArray]) -> [MLXArray])?
        let compiledAnalysisStep: (@Sendable ([MLXArray]) -> [MLXArray])?
        let compiledFeatureStep: (@Sendable ([MLXArray]) -> [MLXArray])?
        let compiledSynthesisStep: (@Sendable ([MLXArray]) -> [MLXArray])?
        let compiledStreamErbDecoderStep: (@Sendable ([MLXArray]) -> [MLXArray])?
        let compiledStreamDfConvpStep: (@Sendable ([MLXArray]) -> [MLXArray])?
        let compiledStreamInferAssignStep: (@Sendable ([MLXArray]) -> [MLXArray])?

        public init(model: DeepFilterNetModel, config: DeepFilterNetStreamingConfig = DeepFilterNetStreamingConfig()) {
            self.model = model
            self.config = config
            self.enableProfiling = config.enableProfiling
            self.profilingForceEvalPerStage = config.profilingForceEvalPerStage

            self.fftSize = model.config.fftSize
            self.hopSize = model.config.hopSize
            self.freqBins = model.config.freqBins
            self.nbDf = model.config.nbDf
            self.nbErb = model.config.nbErb
            self.dfOrder = model.config.dfOrder
            self.dfLookahead = model.config.dfLookahead
            self.convLookahead = model.config.convLookahead

            let alpha = model.normAlpha()
            self.alphaArray = MLXArray(alpha)
            self.oneMinusAlphaArray = MLXArray(Float(1.0) - alpha)
            self.fftScaleArray = MLXArray(Float(model.config.fftSize))
            self.vorbisWindow = model.vorbisWindow.asType(.float32)
            self.wnormArray = MLXArray(model.wnorm)
            self.inferenceDType = model.inferenceDType
            self.analysisMemCount = max(0, model.config.fftSize - model.config.hopSize)
            self.synthMemCount = max(0, model.config.fftSize - model.config.hopSize)
            self.zeroSpecFrame = MLXArray.zeros([model.config.freqBins, 2], type: Float.self)
            self.zeroSpecLowFrame = zeroSpecFrame[0..<model.config.nbDf, 0...]
            self.zeroSpecLowFrameInference = zeroSpecLowFrame.asType(model.inferenceDType)
            self.zeroMaskFrame = MLXArray.zeros([1, 1, 1, model.config.nbErb], type: Float.self)
            self.zeroMaskFrameInference = zeroMaskFrame.asType(model.inferenceDType)
            self.zeroEncErbFrame = MLXArray.zeros([1, 1, 1, model.config.nbErb], type: Float.self).asType(model.inferenceDType)
            self.zeroEncDfFrame = MLXArray.zeros([1, 2, 1, model.config.nbDf], type: Float.self).asType(model.inferenceDType)
            self.zeroDfConvpFrame = MLXArray.zeros([1, model.config.convCh, 1, model.config.nbDf], type: Float.self).asType(model.inferenceDType)
            self.dfConvpKernelSizeT = max(1, model.config.dfPathwayKernelSizeT)
            let leftHistory = max(0, model.config.dfOrder - model.config.dfLookahead - 1)
            self.dfSpecLeft = leftHistory
            self.specRingCapacity = max(8, leftHistory + model.config.convLookahead + model.config.dfLookahead + 4)
            if model.erbFB.shape.count == 2,
               model.erbFB.shape[0] == model.config.freqBins,
               model.erbFB.shape[1] == model.config.nbErb
            {
                self.erbFBFrame = model.erbFB.asType(.float32)
            } else {
                self.erbFBFrame = nil
            }
            self.lsnrWeight = (try? model.w("enc.lsnr_fc.0.weight")) ?? MLXArray.zeros([1, model.config.embHiddenDim], type: Float.self)
            self.lsnrBias = (try? model.w("enc.lsnr_fc.0.bias")) ?? MLXArray.zeros([1], type: Float.self)
            self.lsnrScale = MLXArray(Float(model.config.lsnrMax - model.config.lsnrMin)).asType(model.inferenceDType)
            self.lsnrOffset = MLXArray(Float(model.config.lsnrMin)).asType(model.inferenceDType)
            self.encDfFcEmbWeight = Self.requireWeight(model, key: "enc.df_fc_emb.0.weight")
            self.dfDecSkipWeight = model.weights["df_dec.df_skip.weight"]
            self.dfDecOutWeight = Self.requireWeight(model, key: "df_dec.df_out.0.weight")
            self.encEmbGRU = Self.buildStreamGRU(model: model, prefix: "enc.emb_gru")
            self.erbDecEmbGRU = Self.buildStreamGRU(model: model, prefix: "erb_dec.emb_gru")
            self.dfDecGRU = Self.buildStreamGRU(model: model, prefix: "df_dec.df_gru")
            self.encErbConv0 = Self.buildStreamConvLayer(
                model: model,
                prefix: "enc.erb_conv0",
                main: 1,
                pointwise: nil,
                bn: 2,
                fstride: 1
            )
            self.encErbConv1 = Self.buildStreamConvLayer(
                model: model,
                prefix: "enc.erb_conv1",
                main: 0,
                pointwise: 1,
                bn: 2,
                fstride: 2
            )
            self.encErbConv2 = Self.buildStreamConvLayer(
                model: model,
                prefix: "enc.erb_conv2",
                main: 0,
                pointwise: 1,
                bn: 2,
                fstride: 2
            )
            self.encErbConv3 = Self.buildStreamConvLayer(
                model: model,
                prefix: "enc.erb_conv3",
                main: 0,
                pointwise: 1,
                bn: 2,
                fstride: 1
            )
            self.encDfConv0 = Self.buildStreamConvLayer(
                model: model,
                prefix: "enc.df_conv0",
                main: 1,
                pointwise: 2,
                bn: 3,
                fstride: 1
            )
            self.encDfConv1 = Self.buildStreamConvLayer(
                model: model,
                prefix: "enc.df_conv1",
                main: 0,
                pointwise: 1,
                bn: 2,
                fstride: 2
            )

            self.analysisMem = MLXArray.zeros([analysisMemCount], type: Float.self)
            self.synthMem = MLXArray.zeros([synthMemCount], type: Float.self)
            self.erbState = MLXArray(Self.linspace(start: -60.0, end: -90.0, count: model.config.nbErb))
            self.dfState = MLXArray(Self.linspace(start: 0.001, end: 0.0001, count: model.config.nbDf))
            self.rings = DeepFilterNetStreamingRings(
                spec: TensorRingBuffer(capacity: specRingCapacity, initial: zeroSpecFrame),
                specLow: TensorRingBuffer(capacity: specRingCapacity, initial: zeroSpecLowFrameInference),
                encErb: TensorRingBuffer(capacity: 3, initial: zeroEncErbFrame),
                encDf: TensorRingBuffer(capacity: 3, initial: zeroEncDfFrame),
                dfConvp: TensorRingBuffer(capacity: max(1, dfConvpKernelSizeT), initial: zeroDfConvpFrame)
            )
            if model.performanceConfig.preferCompiledGraphs {
                if let erbFB = self.erbFBFrame {
                    self.compiledAnalysisFeatureStep = Self.buildCompiledAnalysisFeatureStep(
                        analysisMemCount: self.analysisMemCount,
                        hopSize: self.hopSize,
                        vorbisWindow: self.vorbisWindow,
                        wnormArray: self.wnormArray,
                        nbDf: model.config.nbDf,
                        erbFBFrame: erbFB,
                        tenArray: self.tenArray,
                        epsEnergy: self.epsEnergy,
                        oneMinusAlphaArray: self.oneMinusAlphaArray,
                        alphaArray: self.alphaArray,
                        fortyArray: self.fortyArray,
                        epsNorm: self.epsNorm
                    )
                    self.compiledAnalysisStep = nil
                    self.compiledFeatureStep = Self.buildCompiledFeatureStep(
                        nbDf: model.config.nbDf,
                        erbFBFrame: erbFB,
                        tenArray: self.tenArray,
                        epsEnergy: self.epsEnergy,
                        oneMinusAlphaArray: self.oneMinusAlphaArray,
                        alphaArray: self.alphaArray,
                        fortyArray: self.fortyArray,
                        epsNorm: self.epsNorm
                    )
                } else {
                    self.compiledAnalysisFeatureStep = nil
                    self.compiledAnalysisStep = Self.buildCompiledAnalysisStep(
                        analysisMemCount: self.analysisMemCount,
                        hopSize: self.hopSize,
                        vorbisWindow: self.vorbisWindow,
                        wnormArray: self.wnormArray
                    )
                    self.compiledFeatureStep = nil
                }
                self.compiledSynthesisStep = Self.buildCompiledSynthesisStep(
                    hopSize: self.hopSize,
                    fftSize: self.fftSize,
                    synthMemCount: self.synthMemCount,
                    complexJ: model.j,
                    fftScaleArray: self.fftScaleArray,
                    vorbisWindow: self.vorbisWindow
                )
                self.compiledStreamErbDecoderStep = Self.buildCompiledStreamErbDecoderStep(model: model)
                self.compiledStreamDfConvpStep = Self.buildCompiledStreamDfConvpStep(model: model)
                self.compiledStreamInferAssignStep = Self.buildCompiledStreamInferAssignStep(
                    nbDf: model.config.nbDf,
                    dfOrder: model.config.dfOrder,
                    encConcat: model.config.encConcat
                )
            } else {
                self.compiledAnalysisFeatureStep = nil
                self.compiledAnalysisStep = nil
                self.compiledFeatureStep = nil
                self.compiledSynthesisStep = nil
                self.compiledStreamErbDecoderStep = nil
                self.compiledStreamDfConvpStep = nil
                self.compiledStreamInferAssignStep = nil
            }
        }

        static func buildCompiledAnalysisStep(
            analysisMemCount: Int,
            hopSize: Int,
            vorbisWindow: MLXArray,
            wnormArray: MLXArray
        ) -> @Sendable ([MLXArray]) -> [MLXArray] {
            compile(shapeless: false) { inputs in
                let hop = inputs[0]
                let analysisMem = inputs[1]

                let frame = analysisMemCount > 0
                    ? MLX.concatenated([analysisMem, hop], axis: 0)
                    : hop
                let frameWin = frame * vorbisWindow
                let specComplex = MLXFFT.rfft(frameWin, axis: 0) * wnormArray
                let spec = MLX.stacked([specComplex.realPart(), specComplex.imaginaryPart()], axis: -1)

                let nextAnalysisMem: MLXArray
                if analysisMemCount <= 0 {
                    nextAnalysisMem = MLXArray.zeros([0], type: Float.self)
                } else if analysisMemCount > hopSize {
                    let split = analysisMemCount - hopSize
                    let rotated = MLX.concatenated([
                        analysisMem[hopSize..<analysisMemCount],
                        analysisMem[0..<hopSize],
                    ], axis: 0)
                    nextAnalysisMem = MLX.concatenated([rotated[0..<split], hop], axis: 0)
                } else {
                    nextAnalysisMem = hop[(hopSize - analysisMemCount)..<hopSize]
                }
                return [spec, nextAnalysisMem]
            }
        }

        static func buildCompiledAnalysisFeatureStep(
            analysisMemCount: Int,
            hopSize: Int,
            vorbisWindow: MLXArray,
            wnormArray: MLXArray,
            nbDf: Int,
            erbFBFrame: MLXArray,
            tenArray: MLXArray,
            epsEnergy: MLXArray,
            oneMinusAlphaArray: MLXArray,
            alphaArray: MLXArray,
            fortyArray: MLXArray,
            epsNorm: MLXArray
        ) -> @Sendable ([MLXArray]) -> [MLXArray] {
            compile(shapeless: false) { inputs in
                let hop = inputs[0]
                let analysisMem = inputs[1]
                let erbState = inputs[2]
                let dfState = inputs[3]

                let frame = analysisMemCount > 0
                    ? MLX.concatenated([analysisMem, hop], axis: 0)
                    : hop
                let frameWin = frame * vorbisWindow
                let specComplex = MLXFFT.rfft(frameWin, axis: 0) * wnormArray
                let spec = MLX.stacked([specComplex.realPart(), specComplex.imaginaryPart()], axis: -1)

                let nextAnalysisMem: MLXArray
                if analysisMemCount <= 0 {
                    nextAnalysisMem = MLXArray.zeros([0], type: Float.self)
                } else if analysisMemCount > hopSize {
                    let split = analysisMemCount - hopSize
                    let rotated = MLX.concatenated([
                        analysisMem[hopSize..<analysisMemCount],
                        analysisMem[0..<hopSize],
                    ], axis: 0)
                    nextAnalysisMem = MLX.concatenated([rotated[0..<split], hop], axis: 0)
                } else {
                    nextAnalysisMem = hop[(hopSize - analysisMemCount)..<hopSize]
                }

                let re = spec[0..., 0]
                let im = spec[0..., 1]
                let magSq = re.square() + im.square()
                let erb = MLX.matmul(magSq.expandedDimensions(axis: 0), erbFBFrame).squeezed()
                let erbDB = tenArray * MLX.log10(erb + epsEnergy)
                let nextErbState = erbDB * oneMinusAlphaArray + erbState * alphaArray
                let featErb = (erbDB - nextErbState) / fortyArray

                let dfRe = re[0..<nbDf]
                let dfIm = im[0..<nbDf]
                let mag = MLX.sqrt(dfRe.square() + dfIm.square())
                let nextDfState = mag * oneMinusAlphaArray + dfState * alphaArray
                let denom = MLX.sqrt(MLX.maximum(nextDfState, epsNorm))
                let featDfRe = dfRe / denom
                let featDfIm = dfIm / denom

                let featErbMX = featErb
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
                var featDfMX = MLX.stacked([featDfRe, featDfIm], axis: -1)
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
                featDfMX = featDfMX.transposed(0, 3, 1, 2)

                return [spec, featErbMX, featDfMX, nextAnalysisMem, nextErbState, nextDfState]
            }
        }

        static func buildCompiledFeatureStep(
            nbDf: Int,
            erbFBFrame: MLXArray,
            tenArray: MLXArray,
            epsEnergy: MLXArray,
            oneMinusAlphaArray: MLXArray,
            alphaArray: MLXArray,
            fortyArray: MLXArray,
            epsNorm: MLXArray
        ) -> @Sendable ([MLXArray]) -> [MLXArray] {
            compile(shapeless: false) { inputs in
                let spec = inputs[0]
                let erbState = inputs[1]
                let dfState = inputs[2]

                let re = spec[0..., 0]
                let im = spec[0..., 1]
                let magSq = re.square() + im.square()

                let erb = MLX.matmul(magSq.expandedDimensions(axis: 0), erbFBFrame).squeezed()
                let erbDB = tenArray * MLX.log10(erb + epsEnergy)
                let nextErbState = erbDB * oneMinusAlphaArray + erbState * alphaArray
                let featErb = (erbDB - nextErbState) / fortyArray

                let dfRe = re[0..<nbDf]
                let dfIm = im[0..<nbDf]
                let mag = MLX.sqrt(dfRe.square() + dfIm.square())
                let nextDfState = mag * oneMinusAlphaArray + dfState * alphaArray
                let denom = MLX.sqrt(MLX.maximum(nextDfState, epsNorm))
                let featDfRe = dfRe / denom
                let featDfIm = dfIm / denom

                let featErbMX = featErb
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
                var featDfMX = MLX.stacked([featDfRe, featDfIm], axis: -1)
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
                featDfMX = featDfMX.transposed(0, 3, 1, 2)

                return [featErbMX, featDfMX, nextErbState, nextDfState]
            }
        }

        static func buildCompiledSynthesisStep(
            hopSize: Int,
            fftSize: Int,
            synthMemCount: Int,
            complexJ: MLXArray,
            fftScaleArray: MLXArray,
            vorbisWindow: MLXArray
        ) -> @Sendable ([MLXArray]) -> [MLXArray] {
            compile(shapeless: false) { inputs in
                let specNorm = inputs[0]
                let synthMem = inputs[1]

                let complex = specNorm[0..., 0] + complexJ * specNorm[0..., 1]
                var time = MLXFFT.irfft(complex, axis: 0)
                time = time * fftScaleArray
                time = time * vorbisWindow

                let out = time[0..<hopSize] + synthMem[0..<hopSize]

                let nextSynthMem: MLXArray
                if synthMemCount <= 0 {
                    nextSynthMem = MLXArray.zeros([0], type: Float.self)
                } else {
                    let xSecond = time[hopSize..<fftSize]
                    if synthMemCount > hopSize {
                        let split = synthMemCount - hopSize
                        let rotated = MLX.concatenated([
                            synthMem[hopSize..<synthMemCount],
                            synthMem[0..<hopSize],
                        ], axis: 0)
                        let sFirst = rotated[0..<split] + xSecond[0..<split]
                        let sSecond = xSecond[split..<(split + hopSize)]
                        nextSynthMem = MLX.concatenated([sFirst, sSecond], axis: 0)
                    } else {
                        nextSynthMem = xSecond[0..<synthMemCount]
                    }
                }
                return [out, nextSynthMem]
            }
        }

        static func buildCompiledStreamErbDecoderStep(
            model: DeepFilterNetModel
        ) -> (@Sendable ([MLXArray]) -> [MLXArray])? {
            guard let convt2Dense = try? model.denseTransposeWeight("erb_dec.convt2.0.weight"),
                  let convt1Dense = try? model.denseTransposeWeight("erb_dec.convt1.0.weight"),
                  let conv3pW = try? model.convWeightOHWI("erb_dec.conv3p.0.weight"),
                  let convt3W0 = try? model.convWeightOHWI("erb_dec.convt3.0.weight"),
                  let convt3W1 = try? model.convWeightOHWI("erb_dec.convt3.1.weight"),
                  let conv2pW = try? model.convWeightOHWI("erb_dec.conv2p.0.weight"),
                  let convt2W1 = try? model.convWeightOHWI("erb_dec.convt2.1.weight"),
                  let conv1pW = try? model.convWeightOHWI("erb_dec.conv1p.0.weight"),
                  let convt1W1 = try? model.convWeightOHWI("erb_dec.convt1.1.weight"),
                  let conv0pW = try? model.convWeightOHWI("erb_dec.conv0p.0.weight"),
                  let conv0OutW = try? model.convWeightOHWI("erb_dec.conv0_out.0.weight"),
                  let conv3pBN = try? model.batchNormAffine(prefix: "erb_dec.conv3p.1"),
                  let convt3BN = try? model.batchNormAffine(prefix: "erb_dec.convt3.2"),
                  let conv2pBN = try? model.batchNormAffine(prefix: "erb_dec.conv2p.1"),
                  let convt2BN = try? model.batchNormAffine(prefix: "erb_dec.convt2.2"),
                  let conv1pBN = try? model.batchNormAffine(prefix: "erb_dec.conv1p.1"),
                  let convt1BN = try? model.batchNormAffine(prefix: "erb_dec.convt1.2"),
                  let conv0pBN = try? model.batchNormAffine(prefix: "erb_dec.conv0p.1"),
                  let conv0OutBN = try? model.batchNormAffine(prefix: "erb_dec.conv0_out.1")
            else {
                return nil
            }

            return compile(shapeless: false) { inputs in
                let embDec = inputs[0]
                let e3 = inputs[1]
                let e2 = inputs[2]
                let e1 = inputs[3]
                let e0 = inputs[4]

                let p3 = relu(
                    DeepFilterNetModel.conv2dBCHW(e3, weightOHWI: conv3pW, bias: nil, fstride: 1, lookahead: 0)
                        * conv3pBN.0 + conv3pBN.1
                )
                var d3 = p3 + embDec
                d3 = DeepFilterNetModel.conv2dBCHW(d3, weightOHWI: convt3W0, bias: nil, fstride: 1, lookahead: 0)
                d3 = DeepFilterNetModel.conv2dBCHW(d3, weightOHWI: convt3W1, bias: nil, fstride: 1, lookahead: 0)
                d3 = relu(d3 * convt3BN.0 + convt3BN.1)

                let p2 = relu(
                    DeepFilterNetModel.conv2dBCHW(e2, weightOHWI: conv2pW, bias: nil, fstride: 1, lookahead: 0)
                        * conv2pBN.0 + conv2pBN.1
                )
                var d2 = p2 + d3
                d2 = DeepFilterNetModel.convTranspose2dBCHWDense(d2, denseWeight: convt2Dense, fstride: 2)
                d2 = DeepFilterNetModel.conv2dBCHW(d2, weightOHWI: convt2W1, bias: nil, fstride: 1, lookahead: 0)
                d2 = relu(d2 * convt2BN.0 + convt2BN.1)

                let p1 = relu(
                    DeepFilterNetModel.conv2dBCHW(e1, weightOHWI: conv1pW, bias: nil, fstride: 1, lookahead: 0)
                        * conv1pBN.0 + conv1pBN.1
                )
                var d1 = p1 + d2
                d1 = DeepFilterNetModel.convTranspose2dBCHWDense(d1, denseWeight: convt1Dense, fstride: 2)
                d1 = DeepFilterNetModel.conv2dBCHW(d1, weightOHWI: convt1W1, bias: nil, fstride: 1, lookahead: 0)
                d1 = relu(d1 * convt1BN.0 + convt1BN.1)

                let p0 = relu(
                    DeepFilterNetModel.conv2dBCHW(e0, weightOHWI: conv0pW, bias: nil, fstride: 1, lookahead: 0)
                        * conv0pBN.0 + conv0pBN.1
                )
                let d0 = p0 + d1
                var out = DeepFilterNetModel.conv2dBCHW(d0, weightOHWI: conv0OutW, bias: nil, fstride: 1, lookahead: 0)
                out = out * conv0OutBN.0 + conv0OutBN.1
                return [sigmoid(out)]
            }
        }

        static func buildCompiledStreamDfConvpStep(
            model: DeepFilterNetModel
        ) -> (@Sendable ([MLXArray]) -> [MLXArray])? {
            guard let conv1 = try? model.convWeightOHWI("df_dec.df_convp.1.weight"),
                  let conv2 = try? model.convWeightOHWI("df_dec.df_convp.2.weight"),
                  let bn = try? model.batchNormAffine(prefix: "df_dec.df_convp.3")
            else {
                return nil
            }

            return compile(shapeless: false) { inputs in
                let c0Seq = inputs[0]
                var c0p = DeepFilterNetModel.conv2dBCHW(c0Seq, weightOHWI: conv1, bias: nil, fstride: 1, lookahead: 0)
                c0p = DeepFilterNetModel.conv2dBCHW(c0p, weightOHWI: conv2, bias: nil, fstride: 1, lookahead: 0)
                c0p = relu(c0p * bn.0 + bn.1)
                let t = c0p.shape[2]
                c0p = c0p[0..., 0..., (t - 1)..<t, 0...]
                return [c0p.transposed(0, 2, 3, 1)]
            }
        }

        static func buildCompiledStreamInferAssignStep(
            nbDf: Int,
            dfOrder: Int,
            encConcat: Bool
        ) -> (@Sendable ([MLXArray]) -> [MLXArray]) {
            compile(shapeless: false) { inputs in
                let spec = inputs[0]
                let specMasked = inputs[1]
                let dfCoefsPacked = inputs[2]
                let specLow = inputs[3]

                let coef = dfCoefsPacked
                    .reshaped([1, 1, nbDf, dfOrder, 2])
                    .transposed(0, 3, 1, 2, 4)[0, 0..., 0, 0..<nbDf, 0...]

                let sr = specLow[0..., 0..., 0]
                let si = specLow[0..., 0..., 1]
                let cr = coef[0..., 0..., 0]
                let ci = coef[0..., 0..., 1]

                let outReal = MLX.sum(sr * cr - si * ci, axis: 0)
                let outImag = MLX.sum(sr * ci + si * cr, axis: 0)
                let low = MLX.stacked([outReal, outImag], axis: -1)
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)

                let out: MLXArray
                if encConcat {
                    let high = specMasked[0..., 0..., 0..., nbDf..., 0...]
                    out = MLX.concatenated([low, high], axis: 3)
                } else {
                    let highUnmasked = spec[0..., 0..., 0..., nbDf..., 0...]
                    let specDf = MLX.concatenated([low, highUnmasked], axis: 3)
                    let lowAssigned = specDf[0..., 0..., 0..., 0..<nbDf, 0...]
                    let highMasked = specMasked[0..., 0..., 0..., nbDf..., 0...]
                    out = MLX.concatenated([lowAssigned, highMasked], axis: 3)
                }
                return [out]
            }
        }

        static func requireWeight(_ model: DeepFilterNetModel, key: String) -> MLXArray {
            guard let weight = model.weights[key] else {
                preconditionFailure("Missing required DeepFilterNet weight: \(key)")
            }
            return weight
        }

        static func buildStreamGRU(model: DeepFilterNetModel, prefix: String) -> StreamGRU {
            let linearIn = requireWeight(model, key: "\(prefix).linear_in.0.weight")
            let linearOut = model.weights["\(prefix).linear_out.0.weight"]

            var layers = [StreamGRULayer]()
            var layer = 0
            while model.weights["\(prefix).gru.weight_ih_l\(layer)"] != nil {
                let wihKey = "\(prefix).gru.weight_ih_l\(layer)"
                let whhKey = "\(prefix).gru.weight_hh_l\(layer)"
                let wihT = model.gruTransposedWeights[wihKey] ?? requireWeight(model, key: wihKey).transposed()
                let whhT = model.gruTransposedWeights[whhKey] ?? requireWeight(model, key: whhKey).transposed()
                let bih = requireWeight(model, key: "\(prefix).gru.bias_ih_l\(layer)")
                let bhh = requireWeight(model, key: "\(prefix).gru.bias_hh_l\(layer)")
                layers.append(StreamGRULayer(wihT: wihT, whhT: whhT, bih: bih, bhh: bhh))
                layer += 1
            }

            return StreamGRU(linearInWeight: linearIn, layers: layers, linearOutWeight: linearOut)
        }

        static func convWeightOHWI(model: DeepFilterNetModel, key: String) -> MLXArray {
            if let cached = model.conv2dWeightsOHWI[key] {
                return cached
            }
            return requireWeight(model, key: key).transposed(0, 2, 3, 1)
        }

        static func bnAffine(model: DeepFilterNetModel, prefix: String) -> (MLXArray, MLXArray) {
            if let scale = model.bnScale[prefix], let bias = model.bnBias[prefix] {
                return (scale, bias)
            }
            let gamma = requireWeight(model, key: "\(prefix).weight")
            let beta = requireWeight(model, key: "\(prefix).bias")
            let mean = requireWeight(model, key: "\(prefix).running_mean")
            let variance = requireWeight(model, key: "\(prefix).running_var")
            let scale = (gamma / MLX.sqrt(variance + MLXArray(Float(1e-5))))
                .reshaped([1, gamma.shape[0], 1, 1])
            let shift = (beta - mean * (gamma / MLX.sqrt(variance + MLXArray(Float(1e-5)))))
                .reshaped([1, beta.shape[0], 1, 1])
            return (scale, shift)
        }

        static func buildStreamConvLayer(
            model: DeepFilterNetModel,
            prefix: String,
            main: Int,
            pointwise: Int?,
            bn: Int,
            fstride: Int
        ) -> StreamConvLayer {
            let mainWeight = convWeightOHWI(model: model, key: "\(prefix).\(main).weight")
            let pointwiseWeight = pointwise.map { convWeightOHWI(model: model, key: "\(prefix).\($0).weight") }
            let (bnScale, bnBias) = bnAffine(model: model, prefix: "\(prefix).\(bn)")
            return StreamConvLayer(
                mainWeightOHWI: mainWeight,
                pointwiseWeightOHWI: pointwiseWeight,
                bnScale: bnScale,
                bnBias: bnBias,
                fstride: fstride
            )
        }

        public func reset() {
            pendingSamples = MLXArray.zeros([0], type: Float.self)
            analysisMem = MLXArray.zeros([analysisMemCount], type: Float.self)
            synthMem = MLXArray.zeros([synthMemCount], type: Float.self)
            erbState = MLXArray(Self.linspace(start: -60.0, end: -90.0, count: nbErb))
            dfState = MLXArray(Self.linspace(start: 0.001, end: 0.0001, count: nbDf))
            rings.reset()
            recurrentState.reset()
            delayDropped = 0
            hopsSinceMaterialize = 0
            profHopCount = 0
            profAnalysisSeconds = 0.0
            profFeaturesSeconds = 0.0
            profInferSeconds = 0.0
            profInferEncodeSeconds = 0.0
            profInferEmbSeconds = 0.0
            profInferErbSeconds = 0.0
            profInferDfSeconds = 0.0
            profSynthesisSeconds = 0.0
            profMaterializeSeconds = 0.0
        }

        public func processChunk(_ chunk: MLXArray, isLast: Bool = false) throws -> MLXArray {
            guard chunk.ndim == 1 else {
                throw DeepFilterNetError.invalidAudioShape(chunk.shape)
            }
            let chunkF32 = chunk.asType(.float32)
            if !isLast, pendingSamples.shape[0] == 0, chunkF32.shape[0] == hopSize {
                if var out = try processHop(chunkF32) {
                    if config.compensateDelay {
                        let totalDelay = fftSize - hopSize
                        if delayDropped < totalDelay {
                            let toDrop = min(totalDelay - delayDropped, out.shape[0])
                            if toDrop > 0 {
                                out = out[toDrop..<out.shape[0]]
                                delayDropped += toDrop
                            }
                        }
                    }
                    return out
                }
                return MLXArray.zeros([0], type: Float.self)
            }
            if chunkF32.shape[0] > 0 {
                if pendingSamples.shape[0] == 0 {
                    pendingSamples = chunkF32
                } else {
                    pendingSamples = MLX.concatenated([pendingSamples, chunkF32], axis: 0)
                }
            }

            var outs = [MLXArray]()
            while pendingSamples.shape[0] >= hopSize {
                let hop = pendingSamples[0..<hopSize]
                pendingSamples = pendingSamples[hopSize..<pendingSamples.shape[0]]
                if let out = try processHop(hop) {
                    outs.append(out)
                }
            }

            if isLast {
                if config.padEndFrames > 0 {
                    let pad = MLXArray.zeros([config.padEndFrames * hopSize], type: Float.self)
                    if pendingSamples.shape[0] == 0 {
                        pendingSamples = pad
                    } else {
                        pendingSamples = MLX.concatenated([pendingSamples, pad], axis: 0)
                    }
                }
                while pendingSamples.shape[0] >= hopSize {
                    let hop = pendingSamples[0..<hopSize]
                    pendingSamples = pendingSamples[hopSize..<pendingSamples.shape[0]]
                    if let out = try processHop(hop) {
                        outs.append(out)
                    }
                }
            }

            var y: MLXArray
            if outs.isEmpty {
                y = MLXArray.zeros([0], type: Float.self)
            } else if outs.count == 1, let first = outs.first {
                y = first
            } else {
                y = MLX.concatenated(outs, axis: 0)
            }

            if config.compensateDelay {
                let totalDelay = fftSize - hopSize
                if delayDropped < totalDelay {
                    let toDrop = min(totalDelay - delayDropped, y.shape[0])
                    if toDrop > 0 {
                        y = y[toDrop..<y.shape[0]]
                        delayDropped += toDrop
                    }
                }
            }

            return y
        }

        public func processChunk(_ chunk: [Float], isLast: Bool = false) throws -> [Float] {
            guard !chunk.isEmpty || isLast else { return [] }
            let y = try processChunk(MLXArray(chunk), isLast: isLast)
            if y.shape[0] == 0 {
                return []
            }
            return y.asArray(Float.self)
        }

        public func flush() throws -> [Float] {
            try processChunk([], isLast: true)
        }

        public func flushMLX() throws -> MLXArray {
            try processChunk(MLXArray.zeros([0], type: Float.self), isLast: true)
        }

        public func profilingSummary() -> String? {
            guard enableProfiling else { return nil }
            let hops = max(profHopCount, 1)
            let total = profAnalysisSeconds + profFeaturesSeconds + profInferSeconds + profSynthesisSeconds + profMaterializeSeconds
            let perHopMs = (total / Double(hops)) * 1000.0
            func pct(_ v: Double) -> Double {
                guard total > 0 else { return 0.0 }
                return (v / total) * 100.0
            }
            return String(
                format:
                    """
                    Stream profile: hops=%d total=%.3fs perHop=%.3fms
                      analysis:    %.3fs (%.1f%%)
                      features:    %.3fs (%.1f%%)
                      infer:       %.3fs (%.1f%%)
                        infer.enc: %.3fs (%.1f%% infer)
                        infer.emb: %.3fs (%.1f%% infer)
                        infer.erb: %.3fs (%.1f%% infer)
                        infer.df:  %.3fs (%.1f%% infer)
                      synthesis:   %.3fs (%.1f%%)
                      materialize: %.3fs (%.1f%%)
                    """,
                profHopCount,
                total,
                perHopMs,
                profAnalysisSeconds, pct(profAnalysisSeconds),
                profFeaturesSeconds, pct(profFeaturesSeconds),
                profInferSeconds, pct(profInferSeconds),
                profInferEncodeSeconds, profInferSeconds > 0 ? (profInferEncodeSeconds / profInferSeconds) * 100.0 : 0.0,
                profInferEmbSeconds, profInferSeconds > 0 ? (profInferEmbSeconds / profInferSeconds) * 100.0 : 0.0,
                profInferErbSeconds, profInferSeconds > 0 ? (profInferErbSeconds / profInferSeconds) * 100.0 : 0.0,
                profInferDfSeconds, profInferSeconds > 0 ? (profInferDfSeconds / profInferSeconds) * 100.0 : 0.0,
                profSynthesisSeconds, pct(profSynthesisSeconds),
                profMaterializeSeconds, pct(profMaterializeSeconds)
            )
        }

        func processHop(_ hopTD: MLXArray) throws -> MLXArray? {
            let tAnalysis0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
            let spec: MLXArray
            let featErb: MLXArray
            let featDf: MLXArray
            if let compiledAnalysisFeatureStep {
                let out = compiledAnalysisFeatureStep([hopTD, analysisMem, erbState, dfState])
                spec = out[0]
                featErb = out[1]
                featDf = out[2]
                analysisMem = out[3]
                erbState = out[4]
                dfState = out[5]
            } else {
                spec = analysisFrame(hopTD)
                if enableProfiling, profilingForceEvalPerStage {
                    eval(spec)
                }
                if enableProfiling {
                    profAnalysisSeconds += CFAbsoluteTimeGetCurrent() - tAnalysis0
                }

                let tFeatures0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
                let ff = featuresFrame(spec)
                featErb = ff.0
                featDf = ff.1
                if enableProfiling, profilingForceEvalPerStage {
                    eval(featErb, featDf)
                }
                if enableProfiling {
                    profFeaturesSeconds += CFAbsoluteTimeGetCurrent() - tFeatures0
                }
            }
            if enableProfiling, compiledAnalysisFeatureStep != nil {
                let dt = CFAbsoluteTimeGetCurrent() - tAnalysis0
                // Combined path; attribute to analysis bucket for continuity.
                profAnalysisSeconds += dt
            }
            rings.spec.push(spec)
            rings.specLow.push(spec[0..<nbDf, 0...].asType(inferenceDType))
            rings.encErb.push(featErb.asType(inferenceDType))
            rings.encDf.push(featDf.asType(inferenceDType))

            if rings.spec.totalWritten <= convLookahead {
                return nil
            }
            let targetFrameIndex = rings.spec.totalWritten - 1 - convLookahead
            guard let specT = rings.spec.get(absoluteIndex: targetFrameIndex) else {
                return nil
            }
            let dfSpecHistory = dfSpecHistoryForTargetFrame(targetFrameIndex)
            let tInfer0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
            let specEnhanced = try inferFrame(
                spec: specT,
                targetFrameIndex: targetFrameIndex,
                dfSpecHistory: dfSpecHistory
            )
            if enableProfiling, profilingForceEvalPerStage {
                eval(specEnhanced)
            }
            if enableProfiling {
                profInferSeconds += CFAbsoluteTimeGetCurrent() - tInfer0
            }

            let tSynth0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
            let out = synthesisFrame(specEnhanced.asType(.float32))
            if enableProfiling, profilingForceEvalPerStage {
                eval(out)
            }
            if enableProfiling {
                profSynthesisSeconds += CFAbsoluteTimeGetCurrent() - tSynth0
            }
            hopsSinceMaterialize += 1
            if config.materializeEveryHops > 0, hopsSinceMaterialize >= config.materializeEveryHops {
                let tMat0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
                materializeStreamingState(output: out)
                if enableProfiling {
                    profMaterializeSeconds += CFAbsoluteTimeGetCurrent() - tMat0
                }
                hopsSinceMaterialize = 0
            }
            if enableProfiling {
                profHopCount += 1
            }
            return out
        }

        func analysisFrame(_ hopTD: MLXArray) -> MLXArray {
            if let compiledAnalysisStep {
                let out = compiledAnalysisStep([hopTD, analysisMem])
                analysisMem = out[1]
                return out[0]
            }

            let frame = analysisMemCount > 0
                ? MLX.concatenated([analysisMem, hopTD], axis: 0)
                : hopTD
            let frameWin = frame * vorbisWindow
            let specComplex = MLXFFT.rfft(frameWin, axis: 0) * wnormArray
            let spec = MLX.stacked([specComplex.realPart(), specComplex.imaginaryPart()], axis: -1)
            updateAnalysisMemory(with: hopTD)
            return spec
        }

        func synthesisFrame(_ specNorm: MLXArray) -> MLXArray {
            if let compiledSynthesisStep {
                let out = compiledSynthesisStep([specNorm, synthMem])
                synthMem = out[1]
                return out[0]
            }

            let complex = specNorm[0..., 0] + model.j * specNorm[0..., 1]
            var time = MLXFFT.irfft(complex, axis: 0)
            time = time * fftScaleArray
            time = time * vorbisWindow

            let out = time[0..<hopSize] + synthMem[0..<hopSize]
            updateSynthesisMemory(with: time)
            return out
        }

        func updateAnalysisMemory(with hop: MLXArray) {
            guard analysisMemCount > 0 else { return }
            if analysisMemCount > hopSize {
                let split = analysisMemCount - hopSize
                let rotated = MLX.concatenated([
                    analysisMem[hopSize..<analysisMemCount],
                    analysisMem[0..<hopSize],
                ], axis: 0)
                analysisMem = MLX.concatenated([rotated[0..<split], hop], axis: 0)
            } else {
                analysisMem = hop[(hopSize - analysisMemCount)..<hopSize]
            }
        }

        func updateSynthesisMemory(with time: MLXArray) {
            guard synthMemCount > 0 else { return }
            let xSecond = time[hopSize..<fftSize]
            if synthMemCount > hopSize {
                let split = synthMemCount - hopSize
                let rotated = MLX.concatenated([
                    synthMem[hopSize..<synthMemCount],
                    synthMem[0..<hopSize],
                ], axis: 0)
                let sFirst = rotated[0..<split] + xSecond[0..<split]
                let sSecond = xSecond[split..<(split + hopSize)]
                synthMem = MLX.concatenated([sFirst, sSecond], axis: 0)
            } else {
                synthMem = xSecond[0..<synthMemCount]
            }
        }

        func featuresFrame(_ spec: MLXArray) -> (MLXArray, MLXArray) {
            if let compiledFeatureStep {
                let out = compiledFeatureStep([spec, erbState, dfState])
                erbState = out[2]
                dfState = out[3]
                return (out[0], out[1])
            }

            let re = spec[0..., 0]
            let im = spec[0..., 1]
            let magSq = re.square() + im.square()

            let erb: MLXArray
            if let erbFBFrame {
                erb = MLX.matmul(magSq.expandedDimensions(axis: 0), erbFBFrame).squeezed()
            } else {
                var erbBands = [MLXArray]()
                erbBands.reserveCapacity(nbErb)
                var start = 0
                for width in model.erbBandWidths {
                    let stop = min(start + width, freqBins)
                    if stop > start {
                        erbBands.append(MLX.mean(magSq[start..<stop], axis: 0))
                    } else {
                        erbBands.append(MLXArray.zeros([1], type: Float.self).squeezed())
                    }
                    start = stop
                }
                erb = MLX.stacked(erbBands, axis: 0)
            }
            let erbDB = tenArray * MLX.log10(erb + epsEnergy)
            erbState = erbDB * oneMinusAlphaArray + erbState * alphaArray
            let featErb = (erbDB - erbState) / fortyArray

            let dfRe = re[0..<nbDf]
            let dfIm = im[0..<nbDf]
            let mag = MLX.sqrt(dfRe.square() + dfIm.square())
            dfState = mag * oneMinusAlphaArray + dfState * alphaArray
            let denom = MLX.sqrt(MLX.maximum(dfState, epsNorm))
            let featDfRe = dfRe / denom
            let featDfIm = dfIm / denom

            let featErbMX = featErb
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
            var featDfMX = MLX.stacked([featDfRe, featDfIm], axis: -1)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
            featDfMX = featDfMX.transposed(0, 3, 1, 2)
            return (featErbMX, featDfMX)
        }

        func inferFrame(
            spec: MLXArray,
            targetFrameIndex: Int,
            dfSpecHistory: MLXArray
        ) throws -> MLXArray {
            let tEncode0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
            let specMX = spec
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
                .expandedDimensions(axis: 0)
                .asType(inferenceDType)

            let encErbSeq = historyTensor(
                from: rings.encErb,
                requiredCount: 3,
                zeroFrame: zeroEncErbFrame
            )
            let encDfSeq = historyTensor(
                from: rings.encDf,
                requiredCount: 3,
                zeroFrame: zeroEncDfFrame
            )

            let e0 = applyConvLast(input: encErbSeq, layer: encErbConv0)
            let e1 = applyConvLast(input: e0, layer: encErbConv1)
            let e2 = applyConvLast(input: e1, layer: encErbConv2)
            let e3 = applyConvLast(input: e2, layer: encErbConv3)

            let c0 = applyConvLast(input: encDfSeq, layer: encDfConv0)
            let c1 = applyConvLast(input: c0, layer: encDfConv1)

            var cemb = c1.transposed(0, 2, 3, 1).reshaped([1, 1, -1])
            cemb = relu(model.groupedLinear(cemb, weight: encDfFcEmbWeight))

            var emb = e3.transposed(0, 2, 3, 1).reshaped([1, 1, -1])
            emb = model.config.encConcat ? MLX.concatenated([emb, cemb], axis: -1) : (emb + cemb)
            if enableProfiling, profilingForceEvalPerStage {
                eval(e3, c1, emb)
            }
            if enableProfiling {
                profInferEncodeSeconds += CFAbsoluteTimeGetCurrent() - tEncode0
            }

            let tEmb0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
            emb = squeezedGRUStep(
                emb,
                gru: encEmbGRU,
                hiddenSize: model.config.embHiddenDim,
                state: &recurrentState.encEmb
            )

            let applyGains: Bool
            let applyGainZeros: Bool
            let applyDf: Bool
            if config.enableStageSkipping {
                let lsnr = sigmoid(model.linear(emb, weight: lsnrWeight, bias: lsnrBias))
                    * lsnrScale.asType(emb.dtype) + lsnrOffset.asType(emb.dtype)
                let lsnrValue = lsnr.asArray(Float.self).first ?? Float(model.config.lsnrMin)
                (applyGains, applyGainZeros, applyDf) = applyStages(lsnr: lsnrValue)
            } else {
                (applyGains, applyGainZeros, applyDf) = (true, false, true)
            }
            if enableProfiling, profilingForceEvalPerStage {
                eval(emb)
            }
            if enableProfiling {
                profInferEmbSeconds += CFAbsoluteTimeGetCurrent() - tEmb0
            }

            let mask: MLXArray
            let tErb0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
            if applyGains {
                mask = try erbDecoderStep(emb: emb, e3: e3, e2: e2, e1: e1, e0: e0)
            } else if applyGainZeros {
                mask = zeroMaskFrameInference
            } else {
                return specMX[0, 0, 0, 0..., 0...]
            }
            let specMasked = applyMask(spec: specMX, mask: mask)
            if enableProfiling, profilingForceEvalPerStage {
                eval(mask, specMasked)
            }
            if enableProfiling {
                profInferErbSeconds += CFAbsoluteTimeGetCurrent() - tErb0
            }
            if !applyDf {
                return specMasked[0, 0, 0, 0..., 0...]
            }

            let tDf0 = enableProfiling ? CFAbsoluteTimeGetCurrent() : 0
            let dfCoefsPacked = try dfDecoderStep(emb: emb, c0: c0)

            let specEnhanced = try deepFilterAssignPacked(
                spec: specMX,
                specMasked: specMasked,
                dfCoefsPacked: dfCoefsPacked,
                targetFrameIndex: targetFrameIndex,
                dfSpecHistory: dfSpecHistory
            )
            if enableProfiling, profilingForceEvalPerStage {
                eval(dfCoefsPacked, specEnhanced)
            }
            if enableProfiling {
                profInferDfSeconds += CFAbsoluteTimeGetCurrent() - tDf0
            }
            return specEnhanced[0, 0, 0, 0..., 0...]
        }

        func applyStages(lsnr: Float) -> (Bool, Bool, Bool) {
            if lsnr < config.minDbThresh {
                // Only noise detected: apply zero ERB mask and skip DF.
                return (false, true, false)
            }
            if lsnr > config.maxDbErbThresh {
                // Clean speech detected: skip ERB and DF.
                return (false, false, false)
            }
            if lsnr > config.maxDbDfThresh {
                // Mild noise: apply ERB gains only.
                return (true, false, false)
            }
            // Regular noisy speech: apply both stages.
            return (true, false, true)
        }

        func applyConvLast(input: MLXArray, layer: StreamConvLayer) -> MLXArray {
            var y = input
            y = model.conv2dLayer(
                y,
                weightOHWI: layer.mainWeightOHWI,
                bias: nil,
                fstride: layer.fstride,
                lookahead: 0
            )
            if let pointwiseWeightOHWI = layer.pointwiseWeightOHWI {
                y = model.conv2dLayer(
                    y,
                    weightOHWI: pointwiseWeightOHWI,
                    bias: nil,
                    fstride: 1,
                    lookahead: 0
                )
            }
            y = y * layer.bnScale + layer.bnBias
            y = relu(y)
            let t = y.shape[2]
            return y[0..., 0..., (t - 1)..<t, 0...]
        }

        func squeezedGRUStep(
            _ x: MLXArray,
            gru: StreamGRU,
            hiddenSize: Int,
            state: inout [MLXArray]?
        ) -> MLXArray {
            var y = relu(model.groupedLinear(x, weight: gru.linearInWeight))
            var nextState = [MLXArray]()
            nextState.reserveCapacity(gru.layers.count)
            for (layer, layerDef) in gru.layers.enumerated() {
                let prevState: MLXArray
                if let state, layer < state.count {
                    prevState = state[layer]
                } else {
                    prevState = MLXArray.zeros([y.shape[0], hiddenSize], type: Float.self)
                }
                let h = gruLayerStep(y, layer: layerDef, hiddenSize: hiddenSize, prevState: prevState)
                nextState.append(h)
                y = h.expandedDimensions(axis: 1)
            }

            state = nextState
            if let linearOut = gru.linearOutWeight {
                y = relu(model.groupedLinear(y, weight: linearOut))
            }
            return y
        }

        func gruLayerStep(
            _ x: MLXArray,
            layer: StreamGRULayer,
            hiddenSize: Int,
            prevState: MLXArray
        ) -> MLXArray {
            let xt = x[0..., 0, 0...]

            if model.performanceConfig.enableMetalFusedGRUGates,
               xt.dtype == prevState.dtype,
               (xt.dtype == .float32 || xt.dtype == .float16)
            {
                let gx = MLX.addMM(layer.bih, xt, layer.wihT)
                let gh = MLX.addMM(layer.bhh, prevState, layer.whhT)
                if let fused = DeepFilterNetKernelFusion.shared.gruGateStep(
                    gx: gx,
                    gh: gh,
                    prev: prevState,
                    hiddenSize: hiddenSize,
                    threadGroupSize: model.performanceConfig.kernelThreadGroupSize
                ) {
                    return fused
                }
            }

            if model.performanceConfig.preferCompiledGraphs {
                let cacheKey = "b\(xt.shape[0]):i\(xt.shape[1]):h\(hiddenSize):\(xt.dtype)"
                let compiled: (@Sendable ([MLXArray]) -> [MLXArray])
                if let cached = layer.compiledStepCache[cacheKey] {
                    compiled = cached
                } else {
                    let wihT = layer.wihT
                    let whhT = layer.whhT
                    let bih = layer.bih
                    let bhh = layer.bhh
                    compiled = compile(shapeless: false) { inputs in
                        let xtIn = inputs[0]
                        let prev = inputs[1]
                        let gx = MLX.addMM(bih, xtIn, wihT)
                        let gh = MLX.addMM(bhh, prev, whhT)

                        let xr = gx[0..., 0..<hiddenSize]
                        let xz = gx[0..., hiddenSize..<(2 * hiddenSize)]
                        let xn = gx[0..., (2 * hiddenSize)...]
                        let hr = gh[0..., 0..<hiddenSize]
                        let hz = gh[0..., hiddenSize..<(2 * hiddenSize)]
                        let hn = gh[0..., (2 * hiddenSize)...]

                        let r = sigmoid(xr + hr)
                        let z = sigmoid(xz + hz)
                        let n = tanh(xn + r * hn)
                        let one = MLXArray(Float(1.0)).asType(xtIn.dtype)
                        let h = (one - z) * n + z * prev
                        return [h]
                    }
                    layer.compiledStepCache[cacheKey] = compiled
                }
                return compiled([xt, prevState])[0]
            }

            let gx = MLX.addMM(layer.bih, xt, layer.wihT)
            let gh = MLX.addMM(layer.bhh, prevState, layer.whhT)

            let xr = gx[0..., 0..<hiddenSize]
            let xz = gx[0..., hiddenSize..<(2 * hiddenSize)]
            let xn = gx[0..., (2 * hiddenSize)...]
            let hr = gh[0..., 0..<hiddenSize]
            let hz = gh[0..., hiddenSize..<(2 * hiddenSize)]
            let hn = gh[0..., (2 * hiddenSize)...]

            let r = sigmoid(xr + hr)
            let z = sigmoid(xz + hz)
            let n = tanh(xn + r * hn)
            let one = MLXArray(Float(1.0)).asType(xt.dtype)
            return (one - z) * n + z * prevState
        }

        func erbDecoderStep(
            emb: MLXArray,
            e3: MLXArray,
            e2: MLXArray,
            e1: MLXArray,
            e0: MLXArray
        ) throws -> MLXArray {
            var embDec = squeezedGRUStep(
                emb,
                gru: erbDecEmbGRU,
                hiddenSize: model.config.embHiddenDim,
                state: &recurrentState.erbDec
            )
            let f8 = e3.shape[3]
            embDec = embDec.reshaped([1, 1, f8, -1]).transposed(0, 3, 1, 2)

            if let compiledStreamErbDecoderStep {
                return compiledStreamErbDecoderStep([embDec, e3, e2, e1, e0])[0]
            }

            var d3 = relu(try model.applyPathwayConv(e3, prefix: "erb_dec.conv3p")) + embDec
            d3 = relu(try model.applyRegularBlock(d3, prefix: "erb_dec.convt3"))
            var d2 = relu(try model.applyPathwayConv(e2, prefix: "erb_dec.conv2p")) + d3
            d2 = relu(try model.applyTransposeBlock(d2, prefix: "erb_dec.convt2", fstride: 2))
            var d1 = relu(try model.applyPathwayConv(e1, prefix: "erb_dec.conv1p")) + d2
            d1 = relu(try model.applyTransposeBlock(d1, prefix: "erb_dec.convt1", fstride: 2))
            let d0 = relu(try model.applyPathwayConv(e0, prefix: "erb_dec.conv0p")) + d1
            let out = try model.applyOutputConv(d0, prefix: "erb_dec.conv0_out")
            return sigmoid(out)
        }

        func dfDecoderStep(emb: MLXArray, c0: MLXArray) throws -> MLXArray {
            var c = squeezedGRUStep(
                emb,
                gru: dfDecGRU,
                hiddenSize: model.config.dfHiddenDim,
                state: &recurrentState.dfDec
            )
            if let dfDecSkipWeight {
                c = c + model.groupedLinear(emb, weight: dfDecSkipWeight)
            }

            rings.dfConvp.push(c0)
            let c0Seq = historyTensor(
                from: rings.dfConvp,
                requiredCount: dfConvpKernelSizeT,
                zeroFrame: zeroDfConvpFrame
            )
            let c0p: MLXArray
            if let compiledStreamDfConvpStep {
                c0p = compiledStreamDfConvpStep([c0Seq])[0]
            } else {
                var c = try model.conv2dLayer(
                    c0Seq,
                    weightKey: "df_dec.df_convp.1.weight",
                    bias: nil,
                    fstride: 1,
                    lookahead: 0
                )
                c = try model.conv2dLayer(
                    c,
                    weightKey: "df_dec.df_convp.2.weight",
                    bias: nil,
                    fstride: 1,
                    lookahead: 0
                )
                c = relu(try model.batchNorm(c, prefix: "df_dec.df_convp.3"))
                let t = c.shape[2]
                c0p = c[0..., 0..., (t - 1)..<t, 0...].transposed(0, 2, 3, 1)
            }

            let dfOut = tanh(model.groupedLinear(c, weight: dfDecOutWeight))
                .reshaped([1, 1, nbDf, dfOrder * 2])
            return dfOut + c0p
        }

        func applyMask(spec: MLXArray, mask: MLXArray) -> MLXArray {
            if model.performanceConfig.enableMetalFusedErbInvMaskApply,
               spec.dtype == mask.dtype,
               (spec.dtype == .float32 || spec.dtype == .float16),
               let fused = DeepFilterNetKernelFusion.shared.applyMaskErbInv(
                   spec: spec,
                   mask: mask,
                   erbInvFB: spec.dtype == .float16 ? model.erbInvFBF16 : model.erbInvFBF32,
                   threadGroupSize: model.performanceConfig.kernelThreadGroupSize
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
                erbInv = model.erbInvFBF16
            } else if spec.dtype == .float32 {
                erbInv = model.erbInvFBF32
            } else {
                erbInv = model.erbInvFB.asType(spec.dtype)
            }
            let gains = MLX.matmul(flat, erbInv).reshaped([b, 1, t, model.config.freqBins, 1])
            if model.performanceConfig.enableMetalFusedMaskMultiply,
               let fused = DeepFilterNetKernelFusion.shared.applyMaskMultiply(
                   spec: spec,
                   gains: gains,
                   threadGroupSize: model.performanceConfig.kernelThreadGroupSize,
                   ensureContiguous: model.performanceConfig.ensureKernelContiguousInputs
               )
            {
                return fused
            }
            return spec * gains
        }

        func deepFilterAssignPacked(
            spec: MLXArray,
            specMasked: MLXArray,
            dfCoefsPacked: MLXArray,
            targetFrameIndex _: Int,
            dfSpecHistory: MLXArray
        ) throws -> MLXArray {
            let specLow = dfSpecHistory
            if let compiledStreamInferAssignStep,
               !config.enableStageSkipping
            {
                return compiledStreamInferAssignStep([spec, specMasked, dfCoefsPacked, specLow])[0]
            }

            let coef = dfCoefsPacked
                .reshaped([1, 1, nbDf, dfOrder, 2])
                .transposed(0, 3, 1, 2, 4)[0, 0..., 0, 0..<nbDf, 0...]

            let low: MLXArray
            if model.performanceConfig.enableMetalFusedStreamingDeepFilter,
               let fused = DeepFilterNetKernelFusion.shared.deepFilterStreamingFramePacked(
                   specLow: specLow,
                   coefsPacked: dfCoefsPacked,
                   threadGroupSize: model.performanceConfig.kernelThreadGroupSize
               )
            {
                low = fused
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
            } else if model.performanceConfig.enableMetalFusedStreamingDeepFilter,
                      let fused = DeepFilterNetKernelFusion.shared.deepFilterStreamingFrame(
                   specLow: specLow,
                   coefs: coef,
                   threadGroupSize: model.performanceConfig.kernelThreadGroupSize,
                   ensureContiguous: model.performanceConfig.ensureKernelContiguousInputs
               )
            {
                low = fused
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
            } else {
                let sr = specLow[0..., 0..., 0]
                let si = specLow[0..., 0..., 1]
                let cr = coef[0..., 0..., 0]
                let ci = coef[0..., 0..., 1]

                let outReal = MLX.sum(sr * cr - si * ci, axis: 0)
                let outImag = MLX.sum(sr * ci + si * cr, axis: 0)

                low = MLX.stacked([outReal, outImag], axis: -1)
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
                    .expandedDimensions(axis: 0)
            }

            if model.config.encConcat {
                let high = specMasked[0..., 0..., 0..., nbDf..., 0...]
                return MLX.concatenated([low, high], axis: 3)
            }

            let highUnmasked = spec[0..., 0..., 0..., nbDf..., 0...]
            let specDf = MLX.concatenated([low, highUnmasked], axis: 3)
            let lowAssigned = specDf[0..., 0..., 0..., 0..<nbDf, 0...]
            let highMasked = specMasked[0..., 0..., 0..., nbDf..., 0...]
            return MLX.concatenated([lowAssigned, highMasked], axis: 3)
        }

        func historyTensor(
            from ring: TensorRingBuffer,
            requiredCount: Int,
            zeroFrame: MLXArray
        ) -> MLXArray {
            let have = ring.orderedLast(requiredCount)
            let missing = max(0, requiredCount - have.count)
            var frames = [MLXArray]()
            frames.reserveCapacity(requiredCount)
            if missing > 0 {
                for _ in 0..<missing {
                    frames.append(zeroFrame)
                }
            }
            frames.append(contentsOf: have)
            guard let first = frames.first else {
                return zeroFrame
            }
            if frames.count == 1 {
                return first
            }
            return MLX.concatenated(frames, axis: 2)
        }

        func dfSpecHistoryForTargetFrame(_ targetFrameIndex: Int) -> MLXArray {
            var frames = [MLXArray]()
            frames.reserveCapacity(dfOrder)
            for k in 0..<dfOrder {
                let absoluteIndex = targetFrameIndex - dfSpecLeft + k
                if let frame = rings.specLow.get(absoluteIndex: absoluteIndex) {
                    frames.append(frame)
                } else {
                    frames.append(zeroSpecLowFrameInference)
                }
            }
            return MLX.stacked(frames, axis: 0)
        }

        // Materialize recurrent tensors each hop so MLX does not keep growing a long lazy graph.
        func materializeStreamingState(output: MLXArray) {
            eval(output, analysisMem, synthMem, erbState, dfState)
            if let encEmb = recurrentState.encEmb {
                for x in encEmb { eval(x) }
            }
            if let erbDec = recurrentState.erbDec {
                for x in erbDec { eval(x) }
            }
            if let dfDec = recurrentState.dfDec {
                for x in dfDec { eval(x) }
            }
        }

        static func linspace(start: Float, end: Float, count: Int) -> [Float] {
            guard count > 1 else { return [start] }
            let step = (end - start) / Float(count - 1)
            return (0..<count).map { start + Float($0) * step }
        }
    }
}
