import Accelerate
import DeepFilterNetMLX
import Foundation
import MLX

// MARK: - CPU + Accelerate Streaming Engine

/// Full reimplementation of DeepFilterNet streaming using Accelerate framework.
/// All computation runs on CPU with no GPU dispatch overhead.
public final class CPUAccelerateEngine: StreamingEngine {
    public let engineType: StreamingEngineType = .cpuAccelerate

    // Config
    private var fftSize = 960
    private var hopSize = 480
    private var freqBins = 481
    private var nbErb = 32
    private var nbDf = 96
    private var dfOrder = 5
    private var dfLookahead = 2
    private var convLookahead = 2
    private var convCh = 64
    private var embHiddenDim = 256
    private var dfHiddenDim = 256
    private var encConcat = false
    private var dfPathwayKernelSizeT = 1

    // FFT
    private var fftSetup: vDSP_DFT_Setup?
    private var ifftSetup: vDSP_DFT_Setup?

    // Windows and constants
    private var vorbisWindow = [Float]()
    private var wnorm: Float = 0

    // DSP state
    private var analysisMem = [Float]()
    private var synthMem = [Float]()
    private var erbState = [Float]()
    private var dfState = [Float]()
    private var alpha: Float = 0
    private var oneMinusAlpha: Float = 0

    // Ring buffers (simple arrays)
    private var specRing = [[Float]]()
    private var specLowRing = [[Float]]()
    private var specRingIdx = 0
    private var specRingCapacity = 10
    private var totalFramesWritten = 0

    // Encoder feature history - stored in BCHW layout per frame
    // ERB: each frame is [1 * nbErb] = [nbErb] (C=1, W=nbErb)
    // DF: each frame is [2 * nbDf] laid out as [ch0_f0..ch0_fN, ch1_f0..ch1_fN] (C=2, W=nbDf)
    private var encErbHistory = [[Float]]() // last 3 frames of [nbErb]
    private var encDfHistory = [[Float]]()  // last 3 frames of [2*nbDf]

    // DF conv pathway history - stored as [convCh * nbDf] in CHW layout
    private var dfConvpHistory = [[Float]]()

    // Recurrent state (GRU hidden states)
    private var encEmbGruStates = [[Float]]()
    private var erbDecGruStates = [[Float]]()
    private var dfDecGruStates = [[Float]]()

    // Delay compensation
    private var delayDropped = 0

    // MARK: - Weight storage (extracted from model as [Float])
    private var erbFB = [Float]()          // [freqBins, nbErb] row-major
    private var erbInvFB = [Float]()       // [nbErb, freqBins] row-major

    // MARK: - Conv Layer: weights stored as [O, I, kH, kW] (PyTorch OIHW format)

    private struct CPUConvLayer {
        let weight: [Float]       // [O, I, kH, kW] flattened row-major (I = inPerGroup for grouped conv)
        let outChannels: Int
        let inChannels: Int       // This is inPerGroup from weight shape (1 for depthwise)
        let kH: Int
        let kW: Int
        let fstride: Int
        let pointwiseWeight: [Float]?  // [O2, O1, 1, 1] if present
        let pointwiseOut: Int
        let bnScale: [Float]  // [C] (after pointwise if present, else after main conv)
        let bnBias: [Float]   // [C]
    }

    private var encErbConvs = [CPUConvLayer]()  // 4 layers
    private var encDfConvs = [CPUConvLayer]()   // 2 layers

    // Decoder conv layers
    private struct CPUDecoderConvBlock {
        let conv0Weight: [Float]   // [O, I, kH, kW]
        let conv0Out: Int
        let conv0In: Int
        let conv0KH: Int
        let conv0KW: Int
        let conv1Weight: [Float]?  // second conv if present
        let conv1Out: Int
        let conv1In: Int
        let conv1KH: Int
        let conv1KW: Int
        let bnScale: [Float]
        let bnBias: [Float]
        let isTranspose: Bool
        let fstride: Int
    }

    private var erbDecConv3p: CPUDecoderConvBlock?
    private var erbDecConvt3: CPUDecoderConvBlock?
    private var erbDecConv2p: CPUDecoderConvBlock?
    private var erbDecConvt2: CPUDecoderConvBlock?
    private var erbDecConv1p: CPUDecoderConvBlock?
    private var erbDecConvt1: CPUDecoderConvBlock?
    private var erbDecConv0p: CPUDecoderConvBlock?
    private var erbDecConv0Out: CPUDecoderConvBlock?

    // DF decoder conv weights
    private var dfConvp1Weight = [Float]()
    private var dfConvp1Shape = [Int]()  // [O, I, kH, kW]
    private var dfConvp2Weight = [Float]()
    private var dfConvp2Shape = [Int]()
    private var dfConvpBnScale = [Float]()
    private var dfConvpBnBias = [Float]()

    // GRU weights
    private struct CPUGRULayer {
        let wihT: [Float]      // [I, 3H] (transposed from [3H, I])
        let whhT: [Float]      // [H, 3H] (transposed from [3H, H])
        let bih: [Float]       // [3H]
        let bhh: [Float]       // [3H]
        let inputSize: Int
        let hiddenSize: Int
    }

    private struct CPUSqueezedGRU {
        let linearInWeight: [Float]  // grouped linear [G, I/G, H/G]
        let linearInGroups: Int
        let linearInInputPerGroup: Int
        let linearInOutputPerGroup: Int
        let layers: [CPUGRULayer]
        let linearOutWeight: [Float]?
        let linearOutGroups: Int
        let linearOutInputPerGroup: Int
        let linearOutOutputPerGroup: Int
    }

    private var encEmbGRU: CPUSqueezedGRU?
    private var erbDecEmbGRU: CPUSqueezedGRU?
    private var dfDecGRU: CPUSqueezedGRU?

    // Linear weights
    private var encDfFcEmbWeight = [Float]() // grouped [G, I/G, O/G]
    private var encDfFcEmbGroups = 1
    private var encDfFcEmbIPG = 0
    private var encDfFcEmbOPG = 0
    private var lsnrWeight = [Float]()    // [outDim, inDim] row-major
    private var lsnrBias = [Float]()      // [outDim]
    private var lsnrOutDim = 0
    private var lsnrInDim = 0
    private var dfDecSkipWeight: [Float]?
    private var dfDecSkipGroups = 1
    private var dfDecSkipIPG = 0
    private var dfDecSkipOPG = 0
    private var dfDecOutWeight = [Float]()  // grouped [G, I/G, O/G]
    private var dfDecOutGroups = 1
    private var dfDecOutIPG = 0
    private var dfDecOutOPG = 0

    public init() {}

    // MARK: - Prepare

    public func prepare(model: DeepFilterNetModel) throws -> Double {
        let t0 = CFAbsoluteTimeGetCurrent()

        // Copy config
        fftSize = model.config.fftSize
        hopSize = model.config.hopSize
        freqBins = model.config.freqBins
        nbErb = model.config.nbErb
        nbDf = model.config.nbDf
        dfOrder = model.config.dfOrder
        dfLookahead = model.config.dfLookahead
        convLookahead = model.config.convLookahead
        convCh = model.config.convCh
        embHiddenDim = model.config.embHiddenDim
        dfHiddenDim = model.config.dfHiddenDim
        encConcat = model.config.encConcat
        dfPathwayKernelSizeT = max(1, model.config.dfPathwayKernelSizeT)

        // Setup FFT
        fftSetup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(fftSize), .FORWARD)
        ifftSetup = vDSP_DFT_zrop_CreateSetup(nil, vDSP_Length(fftSize), .INVERSE)

        // Window
        vorbisWindow = extractFloats(model.vorbisWindow)
        wnorm = model.wnorm

        // EMA alpha
        alpha = model.normAlphaValue
        oneMinusAlpha = 1.0 - alpha

        // ERB filterbank
        erbFB = extractFloats(model.erbFB)
        erbInvFB = extractFloats(model.erbInvFB)

        // Extract encoder conv layers
        encErbConvs = [
            extractConvLayer(model, prefix: "enc.erb_conv0", main: 1, pointwise: nil, bn: 2, fstride: 1),
            extractConvLayer(model, prefix: "enc.erb_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2),
            extractConvLayer(model, prefix: "enc.erb_conv2", main: 0, pointwise: 1, bn: 2, fstride: 2),
            extractConvLayer(model, prefix: "enc.erb_conv3", main: 0, pointwise: 1, bn: 2, fstride: 1),
        ]
        encDfConvs = [
            extractConvLayer(model, prefix: "enc.df_conv0", main: 1, pointwise: 2, bn: 3, fstride: 1),
            extractConvLayer(model, prefix: "enc.df_conv1", main: 0, pointwise: 1, bn: 2, fstride: 2),
        ]

        // Extract GRUs
        encEmbGRU = extractSqueezedGRU(model, prefix: "enc.emb_gru")
        erbDecEmbGRU = extractSqueezedGRU(model, prefix: "erb_dec.emb_gru")
        dfDecGRU = extractSqueezedGRU(model, prefix: "df_dec.df_gru")

        // Extract linear weights
        extractGroupedLinear(model, key: "enc.df_fc_emb.0.weight",
                             groups: &encDfFcEmbGroups, ipg: &encDfFcEmbIPG, opg: &encDfFcEmbOPG,
                             weight: &encDfFcEmbWeight)

        if let w = model.weights["df_dec.df_skip.weight"] {
            var g = 1, i = 0, o = 0
            var wt = [Float]()
            extractGroupedLinearFromArray(w, groups: &g, ipg: &i, opg: &o, weight: &wt)
            dfDecSkipWeight = wt
            dfDecSkipGroups = g
            dfDecSkipIPG = i
            dfDecSkipOPG = o
        }

        extractGroupedLinear(model, key: "df_dec.df_out.0.weight",
                             groups: &dfDecOutGroups, ipg: &dfDecOutIPG, opg: &dfDecOutOPG,
                             weight: &dfDecOutWeight)

        let lw = model.weights["enc.lsnr_fc.0.weight"]!.asType(.float32)
        lsnrWeight = extractFloats(lw)
        lsnrOutDim = lw.shape[0]
        lsnrInDim = lw.shape[1]
        lsnrBias = extractFloats(model.weights["enc.lsnr_fc.0.bias"]!)

        // Extract ERB decoder conv layers
        erbDecConv3p = extractPathwayConv(model, prefix: "erb_dec.conv3p")
        erbDecConvt3 = extractRegularBlock(model, prefix: "erb_dec.convt3")
        erbDecConv2p = extractPathwayConv(model, prefix: "erb_dec.conv2p")
        erbDecConvt2 = extractTransposeBlock(model, prefix: "erb_dec.convt2", fstride: 2)
        erbDecConv1p = extractPathwayConv(model, prefix: "erb_dec.conv1p")
        erbDecConvt1 = extractTransposeBlock(model, prefix: "erb_dec.convt1", fstride: 2)
        erbDecConv0p = extractPathwayConv(model, prefix: "erb_dec.conv0p")
        erbDecConv0Out = extractOutputConv(model, prefix: "erb_dec.conv0_out")

        // Extract DF decoder convp weights
        let (c1w, c1s) = extractConvWeight(model, key: "df_dec.df_convp.1.weight")
        dfConvp1Weight = c1w; dfConvp1Shape = c1s
        let (c2w, c2s) = extractConvWeight(model, key: "df_dec.df_convp.2.weight")
        dfConvp2Weight = c2w; dfConvp2Shape = c2s
        let (dbs, dbb) = extractBNAffine(model, prefix: "df_dec.df_convp.3")
        dfConvpBnScale = dbs; dfConvpBnBias = dbb

        // Initialize state
        let dfSpecLeft = max(0, dfOrder - dfLookahead - 1)
        specRingCapacity = max(8, dfSpecLeft + convLookahead + dfLookahead + 4)

        initializeState()

        return CFAbsoluteTimeGetCurrent() - t0
    }

    private func initializeState() {
        let analysisMemCount = max(0, fftSize - hopSize)
        analysisMem = [Float](repeating: 0, count: analysisMemCount)
        synthMem = [Float](repeating: 0, count: analysisMemCount)

        erbState = linspace(start: -60.0, end: -90.0, count: nbErb)
        dfState = linspace(start: 0.001, end: 0.0001, count: nbDf)

        specRing = Array(repeating: [Float](repeating: 0, count: freqBins * 2), count: specRingCapacity)
        specLowRing = Array(repeating: [Float](repeating: 0, count: nbDf * 2), count: specRingCapacity)
        specRingIdx = 0
        totalFramesWritten = 0

        // ERB history: 3 frames of [1 channel * nbErb freq] = [nbErb]
        encErbHistory = Array(repeating: [Float](repeating: 0, count: nbErb), count: 3)
        // DF history: 3 frames of [2 channels * nbDf freq] = [2*nbDf]
        // Layout: [ch0_f0, ch0_f1, ..., ch0_f(nbDf-1), ch1_f0, ..., ch1_f(nbDf-1)]
        encDfHistory = Array(repeating: [Float](repeating: 0, count: 2 * nbDf), count: 3)
        dfConvpHistory = Array(repeating: [Float](repeating: 0, count: convCh * nbDf), count: dfPathwayKernelSizeT)

        encEmbGruStates = []
        erbDecGruStates = []
        dfDecGruStates = []
        delayDropped = 0
    }

    public func reset() {
        initializeState()
    }

    // MARK: - Process Hop

    /// Full neural network inference is enabled.
    private var enableFullInference = true

    public func processHop(_ samples: [Float]) throws -> [Float] {
        guard samples.count == hopSize else { return [] }

        // 1. Analysis: STFT
        let spec = analysisFrame(samples)  // [freqBins * 2] interleaved re,im

        // 2. Features
        let featErb = erbFeatures(spec)     // [nbErb]
        let featDf = dfFeatures(spec)       // [2 * nbDf] laid out as [nbDf re, nbDf im] (C=2, W=nbDf)

        // 3. Push to rings
        pushSpecRing(spec)

        // Push feature history (shift left, append new)
        encErbHistory.removeFirst()
        encErbHistory.append(featErb)
        encDfHistory.removeFirst()
        encDfHistory.append(featDf)

        // 4. Check lookahead
        if totalFramesWritten <= convLookahead {
            return []
        }

        let targetIdx = totalFramesWritten - 1 - convLookahead

        // Get target spec
        let specT = getSpecFromRing(absoluteIndex: targetIdx)

        // 5. Inference
        let specEnhanced: [Float]
        if enableFullInference {
            specEnhanced = try inferFrame(spec: specT, targetFrameIndex: targetIdx)
        } else {
            specEnhanced = specT
        }

        // 6. Synthesis: ISTFT
        var out = synthesisFrame(specEnhanced)

        // 7. Delay compensation
        let totalDelay = fftSize - hopSize
        if delayDropped < totalDelay {
            let toDrop = min(totalDelay - delayDropped, out.count)
            if toDrop > 0 {
                out = Array(out[toDrop...])
                delayDropped += toDrop
            }
        }

        return out
    }

    public func flush() throws -> [Float] {
        var result = [Float]()
        let padFrames = 3
        for _ in 0..<padFrames {
            let out = try processHop([Float](repeating: 0, count: hopSize))
            result.append(contentsOf: out)
        }
        return result
    }

    // MARK: - Analysis (STFT single frame)

    private func analysisFrame(_ hop: [Float]) -> [Float] {
        let analysisMemCount = analysisMem.count

        // Build frame = [analysisMem | hop]
        var frame = [Float](repeating: 0, count: fftSize)
        if analysisMemCount > 0 {
            frame[0..<analysisMemCount] = analysisMem[...]
        }
        frame[analysisMemCount..<(analysisMemCount + hopSize)] = hop[...]

        // Apply window
        vDSP_vmul(frame, 1, vorbisWindow, 1, &frame, 1, vDSP_Length(fftSize))

        // Update analysis memory
        if analysisMemCount > 0 {
            if analysisMemCount > hopSize {
                let split = analysisMemCount - hopSize
                var newMem = [Float](repeating: 0, count: analysisMemCount)
                newMem[0..<split] = analysisMem[hopSize..<analysisMemCount]
                newMem[split..<analysisMemCount] = hop[...]
                analysisMem = newMem
            } else {
                analysisMem = Array(hop[(hopSize - analysisMemCount)...])
            }
        }

        // Real FFT using vDSP
        let halfN = fftSize / 2
        var realPart = [Float](repeating: 0, count: halfN)
        var imagPart = [Float](repeating: 0, count: halfN)

        // Split even/odd for DFT input
        var inputReal = [Float](repeating: 0, count: halfN)
        var inputImag = [Float](repeating: 0, count: halfN)
        for i in 0..<halfN {
            inputReal[i] = frame[2 * i]
            inputImag[i] = frame[2 * i + 1]
        }

        vDSP_DFT_Execute(fftSetup!, inputReal, inputImag, &realPart, &imagPart)

        // Convert to rfft format: DC, bins 1..N/2-1, Nyquist
        var spec = [Float](repeating: 0, count: freqBins * 2)
        let dcReal = realPart[0] + imagPart[0]
        let nyquistReal = realPart[0] - imagPart[0]

        spec[0] = dcReal * wnorm
        spec[1] = 0.0

        let scale = 2.0 * wnorm
        for k in 1..<halfN {
            spec[2 * k] = realPart[k] * scale
            spec[2 * k + 1] = imagPart[k] * scale
        }

        spec[2 * halfN] = nyquistReal * wnorm
        spec[2 * halfN + 1] = 0.0

        return spec
    }

    // MARK: - Features

    private func erbFeatures(_ spec: [Float]) -> [Float] {
        var magSq = [Float](repeating: 0, count: freqBins)
        for f in 0..<freqBins {
            let re = spec[2 * f]
            let im = spec[2 * f + 1]
            magSq[f] = re * re + im * im
        }

        // ERB band energies: magSq @ erbFB -> [nbErb]
        var erb = [Float](repeating: 0, count: nbErb)
        if !erbFB.isEmpty {
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        Int32(freqBins), Int32(nbErb),
                        1.0, erbFB, Int32(nbErb),
                        magSq, 1,
                        0.0, &erb, 1)
        }

        // dB scale + EMA normalization
        var featErb = [Float](repeating: 0, count: nbErb)
        for e in 0..<nbErb {
            let erbDB = 10.0 * log10f(erb[e] + 1e-10)
            erbState[e] = erbDB * oneMinusAlpha + erbState[e] * alpha
            featErb[e] = (erbDB - erbState[e]) / 40.0
        }

        return featErb
    }

    private func dfFeatures(_ spec: [Float]) -> [Float] {
        // Output layout matches BCHW with C=2, W=nbDf:
        // [ch0_f0, ch0_f1, ..., ch0_f(nbDf-1), ch1_f0, ..., ch1_f(nbDf-1)]
        // ch0 = real, ch1 = imaginary
        var featDf = [Float](repeating: 0, count: 2 * nbDf)

        for f in 0..<nbDf {
            let re = spec[2 * f]
            let im = spec[2 * f + 1]
            let mag = sqrtf(re * re + im * im)
            dfState[f] = mag * oneMinusAlpha + dfState[f] * alpha
            let denom = sqrtf(max(dfState[f], 1e-12))
            featDf[f] = re / denom          // Channel 0 (real)
            featDf[nbDf + f] = im / denom   // Channel 1 (imag)
        }

        return featDf
    }

    // MARK: - Ring Buffer

    private func pushSpecRing(_ spec: [Float]) {
        let idx = totalFramesWritten % specRingCapacity
        specRing[idx] = spec

        var specLow = [Float](repeating: 0, count: nbDf * 2)
        specLow[0..<(nbDf * 2)] = spec[0..<(nbDf * 2)]
        specLowRing[idx] = specLow

        totalFramesWritten += 1
    }

    private func getSpecFromRing(absoluteIndex: Int) -> [Float] {
        let oldest = max(0, totalFramesWritten - specRingCapacity)
        guard absoluteIndex >= oldest, absoluteIndex < totalFramesWritten else {
            return [Float](repeating: 0, count: freqBins * 2)
        }
        return specRing[absoluteIndex % specRingCapacity]
    }

    private func getSpecLowFromRing(absoluteIndex: Int) -> [Float] {
        let oldest = max(0, totalFramesWritten - specRingCapacity)
        guard absoluteIndex >= oldest, absoluteIndex < totalFramesWritten else {
            return [Float](repeating: 0, count: nbDf * 2)
        }
        return specLowRing[absoluteIndex % specRingCapacity]
    }

    // MARK: - Inference

    /// Run full neural network inference for a single streaming frame.
    /// Data layout: all intermediate tensors are stored as flat arrays in BCHW order
    /// with B=1, T=1 per frame. A "frame" array of length C*F means
    /// index [c * F + f] for channel c, frequency f.
    private func inferFrame(spec: [Float], targetFrameIndex: Int) throws -> [Float] {
        // === ENCODER ===

        // ERB path: history is 3 frames of [1 * nbErb] in CHW layout
        let e0 = applyEncoderConv(history: encErbHistory, layer: encErbConvs[0], actualInputChannels: 1)
        let e1 = applyEncoderConvChained(e0, layer: encErbConvs[1])
        let e2 = applyEncoderConvChained(e1, layer: encErbConvs[2])
        let e3 = applyEncoderConvChained(e2, layer: encErbConvs[3])

        // DF path: history is 3 frames of [2 * nbDf] in CHW layout
        let c0 = applyEncoderConv(history: encDfHistory, layer: encDfConvs[0], actualInputChannels: 2)
        let c1 = applyEncoderConvChained(c0, layer: encDfConvs[1])

        // === EMBEDDING ===
        // c1 and e3 outputs: each is a ConvResult with outChannels and outFreq.
        // Take the last time frame from each.
        let c1Frame = c1.frames[c1.frames.count - 1]
        let e3Frame = e3.frames[e3.frames.count - 1]

        // Flatten to [F * C] (freq-major) as the MLX code does:
        //   c1.transposed(0,2,3,1).reshaped([B,T,-1]) => [B,T, F*C]
        // In our CHW layout, data is [ch0_f0..ch0_fN, ch1_f0..ch1_fN, ...]
        // Transpose to freq-major: [f0_ch0, f0_ch1, ..., f0_chC, f1_ch0, ...]
        let c1Flat = flattenFreqMajor(c1Frame, channels: c1.channels, freqDim: c1.freqDim)
        let e3Flat = flattenFreqMajor(e3Frame, channels: e3.channels, freqDim: e3.freqDim)

        // cemb = relu(groupedLinear(c1Flat))
        var cemb = cpuGroupedLinearRelu(c1Flat,
                                         weight: encDfFcEmbWeight,
                                         groups: encDfFcEmbGroups,
                                         ipg: encDfFcEmbIPG,
                                         opg: encDfFcEmbOPG)

        // emb = e3Flat + cemb (when encConcat is false)
        var emb: [Float]
        if encConcat {
            emb = e3Flat + cemb  // concatenation
        } else {
            emb = vectorAdd(e3Flat, cemb)
        }

        // Embedding GRU
        emb = cpuSqueezedGRUStep(emb, gru: encEmbGRU!, states: &encEmbGruStates)

        // === ERB DECODER ===
        let mask = erbDecoderStep(emb: emb, e3: e3, e2: e2, e1: e1, e0: e0)
        // Apply ERB mask to get per-frequency gains
        let specMasked = applyErbMask(spec: spec, mask: mask)

        // === DF DECODER ===
        let dfCoefs = dfDecoderStep(emb: emb, c0: c0)
        // === DEEP FILTER ===
        let dfSpecLeft = max(0, dfOrder - dfLookahead - 1)
        var dfSpecHistory = [[Float]]()
        for k in 0..<dfOrder {
            let absIdx = targetFrameIndex - dfSpecLeft + k
            dfSpecHistory.append(getSpecLowFromRing(absoluteIndex: absIdx))
        }

        let specEnhanced = applyDeepFilter(
            spec: spec,
            specMasked: specMasked,
            dfCoefs: dfCoefs,
            dfSpecHistory: dfSpecHistory
        )

        return specEnhanced
    }

    // MARK: - Conv Result

    /// Result of an encoder conv: a list of time frames, each stored as [C * F] in CHW layout.
    private struct ConvResult {
        let frames: [[Float]]
        let channels: Int
        let freqDim: Int
    }

    // MARK: - Encoder Conv2D

    /// Apply a single encoder conv layer to a sequence of 3 time frames.
    ///
    /// Input: 3 frames each of size [actualInputChannels * inFreq] in CHW layout
    /// Weight: [outC, inPerGroup, kH, kW] in OIHW layout
    /// For grouped (depthwise) conv: groups = actualInputChannels / inPerGroup
    /// Time padding: kH-1 on left (causal)
    /// Freq padding: kW/2 on each side
    /// Output: list of time frames, each [outC * outFreq] in CHW layout
    private func applyEncoderConv(history: [[Float]], layer: CPUConvLayer, actualInputChannels: Int) -> ConvResult {
        let kH = layer.kH
        let kW = layer.kW
        let inPerGroup = layer.inChannels  // from weight shape dim 1
        let outC = layer.outChannels
        let fstride = layer.fstride
        let groups = max(1, actualInputChannels / inPerGroup)
        let outPerGroup = outC / groups
        let inFreq = history[0].count / max(1, actualInputChannels)

        // Time padding: kH-1 left pad (causal, no right pad since lookahead=0 for streaming)
        let timePadLeft = kH - 1
        let totalTimeIn = history.count + timePadLeft

        // Freq padding: kW/2 on each side
        let freqPad = kW / 2
        let paddedFreq = inFreq + 2 * freqPad

        // Build padded input: [totalTimeIn][actualInputChannels][paddedFreq]
        let paddedSize = totalTimeIn * actualInputChannels * paddedFreq
        var padded = [Float](repeating: 0, count: paddedSize)

        for t in 0..<history.count {
            let tPadded = t + timePadLeft
            let frame = history[t]
            for c in 0..<actualInputChannels {
                for f in 0..<inFreq {
                    // Input layout: CHW = [c * inFreq + f]
                    let srcIdx = c * inFreq + f
                    // Padded layout: [tPadded][c][f + freqPad]
                    let dstIdx = (tPadded * actualInputChannels + c) * paddedFreq + f + freqPad
                    if srcIdx < frame.count {
                        padded[dstIdx] = frame[srcIdx]
                    }
                }
            }
        }

        // Conv2d output dimensions
        let outTime = totalTimeIn - kH + 1
        let outFreq = (paddedFreq - kW) / fstride + 1

        // Compute grouped conv: output stored as [outTime][outC][outFreq]
        var output = [Float](repeating: 0, count: outTime * outC * outFreq)

        for ot in 0..<outTime {
            for g in 0..<groups {
                for ocg in 0..<outPerGroup {
                    let oc = g * outPerGroup + ocg
                    for of in 0..<outFreq {
                        var sum: Float = 0
                        let fIn = of * fstride
                        for icg in 0..<inPerGroup {
                            let ic = g * inPerGroup + icg
                            for kh in 0..<kH {
                                let tIn = ot + kh
                                for kw in 0..<kW {
                                    // Weight layout: OIHW = [oc][icg][kh][kw]
                                    // where icg indexes within group (0..<inPerGroup)
                                    let wIdx = ((oc * inPerGroup + icg) * kH + kh) * kW + kw
                                    // Padded layout: [tIn][ic][fIn + kw]
                                    let xIdx = (tIn * actualInputChannels + ic) * paddedFreq + fIn + kw
                                    sum += layer.weight[wIdx] * padded[xIdx]
                                }
                            }
                        }
                        // Output layout: [ot][oc][of]
                        output[(ot * outC + oc) * outFreq + of] = sum
                    }
                }
            }
        }

        // Apply pointwise conv if present
        var finalC = outC
        if let pw = layer.pointwiseWeight {
            let pwOut = layer.pointwiseOut
            var pwOutput = [Float](repeating: 0, count: outTime * pwOut * outFreq)
            for ot in 0..<outTime {
                for poc in 0..<pwOut {
                    for of in 0..<outFreq {
                        var sum: Float = 0
                        for ic in 0..<outC {
                            // Pointwise weight: [pwOut, outC, 1, 1] -> index [poc][ic]
                            let wIdx = poc * outC + ic
                            // Input: [ot][ic][of]
                            let xIdx = (ot * outC + ic) * outFreq + of
                            sum += pw[wIdx] * output[xIdx]
                        }
                        pwOutput[(ot * pwOut + poc) * outFreq + of] = sum
                    }
                }
            }
            output = pwOutput
            finalC = pwOut
        }

        // Apply BN + ReLU
        for ot in 0..<outTime {
            for c in 0..<finalC {
                for of in 0..<outFreq {
                    let idx = (ot * finalC + c) * outFreq + of
                    let val = output[idx] * layer.bnScale[c] + layer.bnBias[c]
                    output[idx] = max(0, val)  // ReLU
                }
            }
        }

        // Split into per-frame arrays
        let frameSize = finalC * outFreq
        var result = [[Float]]()
        result.reserveCapacity(outTime)
        for ot in 0..<outTime {
            result.append(Array(output[(ot * frameSize)..<((ot + 1) * frameSize)]))
        }
        return ConvResult(frames: result, channels: finalC, freqDim: outFreq)
    }

    /// Chain encoder convs: take last 3 frames from previous conv output
    private func applyEncoderConvChained(_ input: ConvResult, layer: CPUConvLayer) -> ConvResult {
        let frames = input.frames
        let history: [[Float]]
        if frames.count >= 3 {
            history = Array(frames.suffix(3))
        } else {
            let zeroFrame = [Float](repeating: 0, count: input.channels * input.freqDim)
            let pad = [[Float]](repeating: zeroFrame, count: 3 - frames.count)
            history = pad + frames
        }
        return applyEncoderConv(history: history, layer: layer, actualInputChannels: input.channels)
    }

    // MARK: - Flatten CHW to Freq-Major

    /// Convert from CHW layout [ch0_f0..ch0_fN, ch1_f0..ch1_fN, ...]
    /// to freq-major [f0_ch0, f0_ch1, ..., fN_ch0, fN_ch1, ...]
    /// This matches BCHW -> transposed(0,2,3,1) -> reshaped([B,T,-1])
    private func flattenFreqMajor(_ frame: [Float], channels: Int, freqDim: Int) -> [Float] {
        var result = [Float](repeating: 0, count: channels * freqDim)
        for f in 0..<freqDim {
            for c in 0..<channels {
                // Source: CHW layout => index c * freqDim + f
                // Dest: freq-major => index f * channels + c
                result[f * channels + c] = frame[c * freqDim + f]
            }
        }
        return result
    }

    /// Convert from freq-major [f0_ch0, f0_ch1, ..., fN_ch0, fN_ch1, ...]
    /// back to CHW layout [ch0_f0..ch0_fN, ch1_f0..ch1_fN, ...]
    private func unflattenToCHW(_ flat: [Float], channels: Int, freqDim: Int) -> [Float] {
        var result = [Float](repeating: 0, count: channels * freqDim)
        for f in 0..<freqDim {
            for c in 0..<channels {
                result[c * freqDim + f] = flat[f * channels + c]
            }
        }
        return result
    }

    // MARK: - Vector Ops

    private func vectorAdd(_ a: [Float], _ b: [Float]) -> [Float] {
        let n = min(a.count, b.count)
        var result = [Float](repeating: 0, count: n)
        vDSP_vadd(a, 1, b, 1, &result, 1, vDSP_Length(n))
        return result
    }

    // MARK: - GRU Step

    private func cpuGRUStep(_ x: [Float], layer: CPUGRULayer, prevState: [Float]) -> [Float] {
        let h = layer.hiddenSize
        let inSize = layer.inputSize
        precondition(x.count >= inSize, "GRU: x.count=\(x.count) < inSize=\(inSize)")
        precondition(prevState.count >= h, "GRU: prevState.count=\(prevState.count) < h=\(h)")
        precondition(layer.wihT.count >= inSize * 3 * h, "GRU: wihT.count=\(layer.wihT.count) < \(inSize * 3 * h)")
        precondition(layer.whhT.count >= h * 3 * h, "GRU: whhT.count=\(layer.whhT.count) < \(h * 3 * h)")

        // gx = x @ wihT + bih  => [1, I] @ [I, 3H] -> [1, 3H]
        var gx = [Float](repeating: 0, count: 3 * h)
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    Int32(inSize), Int32(3 * h),
                    1.0, layer.wihT, Int32(3 * h),
                    x, 1,
                    0.0, &gx, 1)
        vDSP_vadd(gx, 1, layer.bih, 1, &gx, 1, vDSP_Length(3 * h))

        // gh = prevState @ whhT + bhh  => [1, H] @ [H, 3H] -> [1, 3H]
        var gh = [Float](repeating: 0, count: 3 * h)
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    Int32(h), Int32(3 * h),
                    1.0, layer.whhT, Int32(3 * h),
                    prevState, 1,
                    0.0, &gh, 1)
        vDSP_vadd(gh, 1, layer.bhh, 1, &gh, 1, vDSP_Length(3 * h))

        // PyTorch GRU gates
        var newState = [Float](repeating: 0, count: h)
        for i in 0..<h {
            let r = 1.0 / (1.0 + expf(-(gx[i] + gh[i])))
            let z = 1.0 / (1.0 + expf(-(gx[h + i] + gh[h + i])))
            let n = tanhf(gx[2 * h + i] + r * gh[2 * h + i])
            newState[i] = (1.0 - z) * n + z * prevState[i]
        }
        return newState
    }

    private func cpuSqueezedGRUStep(_ x: [Float], gru: CPUSqueezedGRU, states: inout [[Float]]) -> [Float] {
        // Linear in + ReLU
        var y = cpuGroupedLinearRelu(x,
                                      weight: gru.linearInWeight,
                                      groups: gru.linearInGroups,
                                      ipg: gru.linearInInputPerGroup,
                                      opg: gru.linearInOutputPerGroup)

        // GRU layers
        var newStates = [[Float]]()
        for (layerIdx, layer) in gru.layers.enumerated() {
            let prevState: [Float]
            if layerIdx < states.count {
                prevState = states[layerIdx]
            } else {
                prevState = [Float](repeating: 0, count: layer.hiddenSize)
            }
            let h = cpuGRUStep(y, layer: layer, prevState: prevState)
            newStates.append(h)
            y = h
        }
        states = newStates

        // Linear out + ReLU (if present)
        if let w = gru.linearOutWeight {
            y = cpuGroupedLinearRelu(y,
                                      weight: w,
                                      groups: gru.linearOutGroups,
                                      ipg: gru.linearOutInputPerGroup,
                                      opg: gru.linearOutOutputPerGroup)
        }

        return y
    }

    // MARK: - Grouped Linear + ReLU

    private func cpuGroupedLinearRelu(_ x: [Float], weight: [Float], groups: Int, ipg: Int, opg: Int) -> [Float] {
        // weight: [G, ipg, opg] flattened row-major
        // y_g = x_g @ w_g  (x_g is [ipg], w_g is [ipg, opg], y_g is [opg])
        precondition(x.count >= groups * ipg, "groupedLinearRelu: x.count=\(x.count) < groups*ipg=\(groups*ipg)")
        precondition(weight.count >= groups * ipg * opg, "groupedLinearRelu: weight.count=\(weight.count) < groups*ipg*opg=\(groups*ipg*opg)")
        var output = [Float](repeating: 0, count: groups * opg)
        for g in 0..<groups {
            let xStart = g * ipg
            let wStart = g * ipg * opg
            let oStart = g * opg
            for o in 0..<opg {
                var sum: Float = 0
                for i in 0..<ipg {
                    sum += x[xStart + i] * weight[wStart + i * opg + o]
                }
                output[oStart + o] = max(0, sum)  // ReLU
            }
        }
        return output
    }

    private func cpuGroupedLinear(_ x: [Float], weight: [Float], groups: Int, ipg: Int, opg: Int) -> [Float] {
        var output = [Float](repeating: 0, count: groups * opg)
        for g in 0..<groups {
            let xStart = g * ipg
            let wStart = g * ipg * opg
            let oStart = g * opg
            for o in 0..<opg {
                var sum: Float = 0
                for i in 0..<ipg {
                    sum += x[xStart + i] * weight[wStart + i * opg + o]
                }
                output[oStart + o] = sum
            }
        }
        return output
    }

    // MARK: - Standard Linear

    /// Compute y = x @ weight^T + bias
    /// weight: [outDim, inDim] row-major, bias: [outDim]
    private func cpuLinear(_ x: [Float], weight: [Float], bias: [Float], outDim: Int, inDim: Int) -> [Float] {
        var output = [Float](repeating: 0, count: outDim)
        for o in 0..<outDim {
            var sum: Float = 0
            for i in 0..<inDim {
                sum += x[i] * weight[o * inDim + i]
            }
            output[o] = sum + bias[o]
        }
        return output
    }

    // MARK: - Conv2D Helper (single time frame, for decoder)

    /// Apply conv2d to a single-frame input in CHW layout with group support.
    /// Input: [inChannels * inFreq] in CHW layout
    /// Weight: [outC, inPerGroup, kH, kW] in OIHW layout
    /// `weightInC` is shape[1] from the weight tensor (= inPerGroup for grouped conv)
    /// groups = inChannels / weightInC
    /// Returns: [outChannels * outFreq] in CHW layout
    private func conv2dSingleFrame(
        input: [Float],
        inChannels: Int,
        inFreq: Int,
        weight: [Float],
        outChannels: Int,
        weightInC: Int,  // shape[1] of weight, = inPerGroup
        kH: Int,
        kW: Int,
        fstride: Int
    ) -> [Float] {
        let inPerGroup = weightInC
        let groups = max(1, inChannels / max(1, inPerGroup))
        let outPerGroup = outChannels / groups
        let freqPad = kW / 2
        let paddedFreq = inFreq + 2 * freqPad

        // Build padded input [inChannels][paddedFreq]
        var padded = [Float](repeating: 0, count: inChannels * paddedFreq)
        for c in 0..<inChannels {
            for f in 0..<inFreq {
                padded[c * paddedFreq + f + freqPad] = input[c * inFreq + f]
            }
        }

        let outFreq = (paddedFreq - kW) / fstride + 1
        var output = [Float](repeating: 0, count: outChannels * outFreq)

        for g in 0..<groups {
            for ocg in 0..<outPerGroup {
                let oc = g * outPerGroup + ocg
                for of in 0..<outFreq {
                    var sum: Float = 0
                    let fIn = of * fstride
                    for icg in 0..<inPerGroup {
                        let ic = g * inPerGroup + icg
                        for kh in 0..<kH {
                            for kw in 0..<kW {
                                // Weight: [oc, icg, kh, kw] - icg is within-group
                                let wIdx = ((oc * inPerGroup + icg) * kH + kh) * kW + kw
                                let xIdx = ic * paddedFreq + fIn + kw
                                sum += weight[wIdx] * padded[xIdx]
                            }
                        }
                    }
                    output[oc * outFreq + of] = sum
                }
            }
        }

        return output
    }

    // MARK: - Depthwise Transpose Conv2D (single frame, for decoder)

    /// Apply depthwise (groups=C) transpose convolution for freq upsampling.
    /// Weight shape: [C, 1, kH, kW] in OIHW format (depthwise: groups=C, inPerGroup=1, outPerGroup=1)
    /// For streaming single frame: kH=1 typically.
    /// fstride=2 means output freq = (inFreq - 1) * 2 + kW - 2*(kW/2)
    /// but for transpose conv the formula is different.
    ///
    /// Actually the transpose conv in the MLX code uses:
    ///   stride=[1, fstride], padding=(kT-1, kF/2), outputPadding=(0, kF/2)
    ///   For kT=1, kF=3, fstride=2: padding=(0, 1), outputPadding=(0, 1)
    ///   Output width = (inWidth - 1) * stride + kF - 2*padding + outputPadding
    ///                = (inWidth - 1) * 2 + 3 - 2 + 1 = 2*inWidth
    private func depthwiseTransposeConv2dSingleFrame(
        input: [Float],
        channels: Int,
        inFreq: Int,
        weight: [Float],
        kH: Int,
        kW: Int,
        fstride: Int
    ) -> (output: [Float], outFreq: Int) {
        // Transpose conv output size:
        // padding = kW/2, outputPadding = kW/2 (from the MLX code)
        // output = (inFreq - 1) * fstride + kW - 2*(kW/2) + (kW/2)
        //        = (inFreq - 1) * fstride + kW - kW/2
        // For kW=3, fstride=2: output = (inFreq-1)*2 + 3 - 1 = 2*inFreq
        // Actually let me be more precise:
        // MLX convTransposed2d: out = (in - 1) * stride - 2*padding + dilation*(kernel-1) + outputPadding + 1
        // With padding=(kT-1, kW/2), outputPadding=(0, kW/2), dilation=1:
        // outW = (inW - 1) * fstride - 2*(kW/2) + (kW-1) + (kW/2) + 1
        //      = (inW - 1) * fstride - kW + 1 + kW - 1 + kW/2 + 1  [integer division]
        // Let me just compute for kW=3, fstride=2:
        //   padding_w = 1, outputPadding_w = 1
        //   outW = (inW-1)*2 - 2*1 + 1*(3-1) + 1 + 1 = 2*inW - 2 - 2 + 2 + 1 + 1 = 2*inW
        // For kW=3, fstride=1:
        //   outW = (inW-1)*1 - 2 + 2 + 1 + 1 = inW + 1... that doesn't seem right
        // Actually the standard formula: outW = (inW - 1) * stride + kW - 2 * pad + output_pad
        //   For kW=3, fs=2, pad=1, opad=1: outW = (inW-1)*2 + 3 - 2 + 1 = 2*inW
        //   For kW=3, fs=1, pad=1, opad=1: outW = (inW-1)*1 + 3 - 2 + 1 = inW + 1...
        // Hmm, but the MLX code only uses transpose conv for fstride=2.
        // Let me just handle fstride=2 case properly.

        let freqPad = kW / 2  // = 1 for kW=3
        let outputPad = kW / 2  // = 1 for kW=3
        let outFreq = (inFreq - 1) * fstride + kW - 2 * freqPad + outputPad
        // For kW=3, fstride=2: outFreq = (inFreq-1)*2 + 3 - 2 + 1 = 2*inFreq

        var output = [Float](repeating: 0, count: channels * outFreq)

        // Depthwise transpose convolution per channel
        // For each input position, scatter-add the weight pattern to output positions
        for c in 0..<channels {
            // For kH=1, just work with kW dimension
            for fi in 0..<inFreq {
                for kw in 0..<kW {
                    let outPos = fi * fstride + kw - freqPad + outputPad / 2
                    // Actually, let me think about this more carefully.
                    // Standard transpose conv: for each input position fi,
                    // the output positions it contributes to are:
                    //   outPos = fi * stride + kw - padding
                    // where kw goes 0..<kW
                    // Then we crop by outputPadding on the right.
                    // Actually outputPadding ADDS to the output, it doesn't crop.
                    //
                    // Let me use a simple approach: compute the full transpose conv
                    // without padding, then crop to the desired size.
                    // Full output size = (inFreq - 1) * stride + kW
                    // Then we crop: skip freqPad on left, keep outFreq values
                    break  // We'll redo this below
                }
                break
            }
            break
        }

        // Redo with scatter approach
        let fullOutFreq = (inFreq - 1) * fstride + kW
        var fullOutput = [Float](repeating: 0, count: channels * fullOutFreq)

        for c in 0..<channels {
            for fi in 0..<inFreq {
                let val = input[c * inFreq + fi]
                for kw in 0..<kW {
                    // Weight: [channels, 1, kH, kW] depthwise, kH=1
                    // For depthwise: weight[c, 0, 0, kw]
                    let wIdx = (c * 1 * kH + 0) * kW + kw
                    let outPos = fi * fstride + kw
                    fullOutput[c * fullOutFreq + outPos] += val * weight[wIdx]
                }
            }
        }

        // Crop: take from freqPad to freqPad + outFreq
        // But we need to figure out the correct cropping.
        // The MLX code uses: padding=(kT-1, kW/2), outputPadding=(0, kW/2)
        // In PyTorch semantics, for ConvTranspose2d:
        //   The full output is (inW-1)*stride + kW
        //   Then padding removes freqPad from each side: fullOutFreq - 2*freqPad
        //   Then outputPadding adds outputPad to the right: fullOutFreq - 2*freqPad + outputPad
        // So: crop left by freqPad, take (fullOutFreq - 2*freqPad + outputPad) values
        // = outFreq (which we computed above)

        var croppedOutput = [Float](repeating: 0, count: channels * outFreq)
        for c in 0..<channels {
            for f in 0..<outFreq {
                let srcF = f + freqPad
                if srcF < fullOutFreq {
                    croppedOutput[c * outFreq + f] = fullOutput[c * fullOutFreq + srcF]
                }
            }
        }

        return (croppedOutput, outFreq)
    }

    // MARK: - Decoder Conv Block (single time frame)

    /// Apply a decoder conv block to a single-frame CHW tensor.
    /// Returns output in CHW layout.
    private func applyDecoderConvBlock(
        input: [Float],
        inChannels: Int,
        inFreq: Int,
        block: CPUDecoderConvBlock,
        activation: Activation = .relu
    ) -> (output: [Float], outChannels: Int, outFreq: Int) {
        var current = input
        var curC = inChannels
        var curF = inFreq

        if block.isTranspose {
            // Transpose conv (depthwise, groups=convCh) followed by pointwise conv
            let (tOut, tOutFreq) = depthwiseTransposeConv2dSingleFrame(
                input: current,
                channels: curC,
                inFreq: curF,
                weight: block.conv0Weight,
                kH: block.conv0KH,
                kW: block.conv0KW,
                fstride: block.fstride
            )
            current = tOut
            curF = tOutFreq
            // conv0Out for transpose conv = outPerGroup for depthwise = 1, so channels stay same
            // curC stays the same (depthwise)

            // Apply pointwise conv (conv1) if present
            if let pw = block.conv1Weight {
                current = conv2dSingleFrame(
                    input: current,
                    inChannels: curC,
                    inFreq: curF,
                    weight: pw,
                    outChannels: block.conv1Out,
                    weightInC: block.conv1In,
                    kH: block.conv1KH,
                    kW: block.conv1KW,
                    fstride: 1
                )
                curC = block.conv1Out
            }
        } else {
            // Regular conv
            current = conv2dSingleFrame(
                input: current,
                inChannels: curC,
                inFreq: curF,
                weight: block.conv0Weight,
                outChannels: block.conv0Out,
                weightInC: block.conv0In,
                kH: block.conv0KH,
                kW: block.conv0KW,
                fstride: block.fstride
            )
            curC = block.conv0Out
            curF = computeConvOutFreq(inFreq: curF, kW: block.conv0KW, fstride: block.fstride)

            // Second conv if present
            if let pw = block.conv1Weight {
                current = conv2dSingleFrame(
                    input: current,
                    inChannels: curC,
                    inFreq: curF,
                    weight: pw,
                    outChannels: block.conv1Out,
                    weightInC: block.conv1In,
                    kH: block.conv1KH,
                    kW: block.conv1KW,
                    fstride: 1
                )
                curC = block.conv1Out
                curF = computeConvOutFreq(inFreq: curF, kW: block.conv1KW, fstride: 1)
            }
        }

        // Apply BN + activation
        for c in 0..<curC {
            for f in 0..<curF {
                let idx = c * curF + f
                var val = current[idx] * block.bnScale[c] + block.bnBias[c]
                switch activation {
                case .relu:
                    val = max(0, val)
                case .sigmoid:
                    val = 1.0 / (1.0 + expf(-val))
                case .none:
                    break
                }
                current[idx] = val
            }
        }

        return (current, curC, curF)
    }

    private enum Activation {
        case relu
        case sigmoid
        case none
    }

    private func computeConvOutFreq(inFreq: Int, kW: Int, fstride: Int) -> Int {
        let freqPad = kW / 2
        let paddedFreq = inFreq + 2 * freqPad
        return (paddedFreq - kW) / fstride + 1
    }

    // MARK: - ERB Decoder

    private func erbDecoderStep(emb: [Float], e3: ConvResult, e2: ConvResult, e1: ConvResult, e0: ConvResult) -> [Float] {
        // ERB decoder embedding GRU
        let embDec = cpuSqueezedGRUStep(emb, gru: erbDecEmbGRU!, states: &erbDecGruStates)

        // Reshape embDec to match e3 spatial dims: [embHiddenDim] -> [C, F] where F = e3.freqDim
        // The MLX code does: embDecStep.reshaped([1, 1, f8, -1]).transposed(0, 3, 1, 2)
        // So it reshapes to [1, 1, F, C] then transposes to [1, C, 1, F]
        // The embDec vector is in flat format (from GRU output).
        // We need to convert it to CHW: first reshape as [F, C] (freq-major),
        // then convert to CHW: [C, F]
        let e3F = e3.freqDim
        let embC = embDec.count / e3F
        // embDec is already in flat format [embHiddenDim].
        // The MLX code reshapes [B,T,embHiddenDim] -> [B,T,F,C] -> transposed to [B,C,T,F]
        // So embDec[f*C + c] maps to CHW index [c*F + f]
        let embDecCHW = unflattenToCHW(embDec, channels: embC, freqDim: e3F)

        // Take last frame from each encoder level
        let e3Last = e3.frames[e3.frames.count - 1]
        let e2Last = e2.frames[e2.frames.count - 1]
        let e1Last = e1.frames[e1.frames.count - 1]
        let e0Last = e0.frames[e0.frames.count - 1]

        // conv3p: pathway conv on e3
        guard let conv3p = erbDecConv3p else {
            return [Float](repeating: 0.5, count: nbErb)
        }
        let (p3, p3C, p3F) = applyDecoderConvBlock(
            input: e3Last,
            inChannels: e3.channels,
            inFreq: e3.freqDim,
            block: conv3p,
            activation: .relu
        )
        // d3 = p3 + embDec
        var d3 = vectorAdd(p3, embDecCHW)
        let d3C = p3C
        var d3F = p3F

        // convt3: regular block (two convs + BN)
        if let convt3 = erbDecConvt3 {
            let (out, outC, outF) = applyDecoderConvBlock(
                input: d3,
                inChannels: d3C,
                inFreq: d3F,
                block: convt3,
                activation: .relu
            )
            d3 = out
            d3F = outF
            _ = outC // should equal d3C
        }

        // conv2p: pathway conv on e2
        guard let conv2p = erbDecConv2p else {
            return [Float](repeating: 0.5, count: nbErb)
        }
        let (p2, _, p2F) = applyDecoderConvBlock(
            input: e2Last,
            inChannels: e2.channels,
            inFreq: e2.freqDim,
            block: conv2p,
            activation: .relu
        )

        // d2 = p2 + d3 (align to min freq)
        let d2MinF = min(p2F, d3F)
        var d2 = [Float](repeating: 0, count: d3C * d2MinF)
        for c in 0..<d3C {
            for f in 0..<d2MinF {
                d2[c * d2MinF + f] = p2[c * p2F + f] + d3[c * d3F + f]
            }
        }
        var d2F = d2MinF

        // convt2: transpose block (freq upsample by 2)
        if let convt2 = erbDecConvt2 {
            let (out, _, outF) = applyDecoderConvBlock(
                input: d2,
                inChannels: d3C,
                inFreq: d2F,
                block: convt2,
                activation: .relu
            )
            d2 = out
            d2F = outF
        }

        // conv1p: pathway conv on e1
        guard let conv1p = erbDecConv1p else {
            return [Float](repeating: 0.5, count: nbErb)
        }
        let (p1, _, p1F) = applyDecoderConvBlock(
            input: e1Last,
            inChannels: e1.channels,
            inFreq: e1.freqDim,
            block: conv1p,
            activation: .relu
        )

        // d1 = p1 + d2
        let d1MinF = min(p1F, d2F)
        var d1 = [Float](repeating: 0, count: d3C * d1MinF)
        for c in 0..<d3C {
            for f in 0..<d1MinF {
                d1[c * d1MinF + f] = p1[c * p1F + f] + d2[c * d2F + f]
            }
        }
        var d1F = d1MinF

        // convt1: transpose block (freq upsample by 2)
        if let convt1 = erbDecConvt1 {
            let (out, _, outF) = applyDecoderConvBlock(
                input: d1,
                inChannels: d3C,
                inFreq: d1F,
                block: convt1,
                activation: .relu
            )
            d1 = out
            d1F = outF
        }

        // conv0p: pathway conv on e0
        guard let conv0p = erbDecConv0p else {
            return [Float](repeating: 0.5, count: nbErb)
        }
        let (p0, _, p0F) = applyDecoderConvBlock(
            input: e0Last,
            inChannels: e0.channels,
            inFreq: e0.freqDim,
            block: conv0p,
            activation: .relu
        )

        // d0 = p0 + d1
        let d0MinF = min(p0F, d1F)
        var d0 = [Float](repeating: 0, count: d3C * d0MinF)
        for c in 0..<d3C {
            for f in 0..<d0MinF {
                d0[c * d0MinF + f] = p0[c * p0F + f] + d1[c * d1F + f]
            }
        }
        let d0F = d0MinF

        // conv0_out: output conv (conv + BN + sigmoid)
        guard let conv0Out = erbDecConv0Out else {
            return [Float](repeating: 0.5, count: nbErb)
        }
        let (mask, _, _) = applyDecoderConvBlock(
            input: d0,
            inChannels: d3C,
            inFreq: d0F,
            block: conv0Out,
            activation: .sigmoid
        )

        // mask should be [1 * nbErb] = [nbErb] (output has 1 channel)
        return mask
    }

    // MARK: - Apply ERB Mask

    private func applyErbMask(spec: [Float], mask: [Float]) -> [Float] {
        // mask: [nbErb] -> gains: mask @ erbInvFB -> [freqBins]
        var gains = [Float](repeating: 0, count: freqBins)
        if !erbInvFB.isEmpty {
            // erbInvFB: [nbErb, freqBins] row-major
            cblas_sgemv(CblasRowMajor, CblasTrans,
                        Int32(nbErb), Int32(freqBins),
                        1.0, erbInvFB, Int32(freqBins),
                        mask, 1,
                        0.0, &gains, 1)
        }

        // Apply gains to spec
        var specMasked = [Float](repeating: 0, count: freqBins * 2)
        for f in 0..<freqBins {
            specMasked[2 * f] = spec[2 * f] * gains[f]
            specMasked[2 * f + 1] = spec[2 * f + 1] * gains[f]
        }
        return specMasked
    }

    // MARK: - DF Decoder

    private func dfDecoderStep(emb: [Float], c0: ConvResult) -> [Float] {
        // DF decoder GRU (no linear_out)
        var c = cpuSqueezedGRUStep(emb, gru: dfDecGRU!, states: &dfDecGruStates)

        // Skip connection
        if let skipW = dfDecSkipWeight {
            let skip = cpuGroupedLinear(emb, weight: skipW, groups: dfDecSkipGroups, ipg: dfDecSkipIPG, opg: dfDecSkipOPG)
            for i in 0..<c.count {
                c[i] += skip[i]
            }
        }

        // DF convp: apply conv chain to c0 history
        // Push c0 last frame to dfConvpHistory
        let c0Last = c0.frames[c0.frames.count - 1]
        dfConvpHistory.removeFirst()
        dfConvpHistory.append(c0Last)

        // Apply two convs + BN + relu on the dfConvpHistory sequence
        let c0p = applyDfConvp()

        // DF output: tanh(grouped_linear(c))
        let dfOutRaw = cpuGroupedLinear(c, weight: dfDecOutWeight, groups: dfDecOutGroups, ipg: dfDecOutIPG, opg: dfDecOutOPG)

        // Apply tanh
        var dfOut = [Float](repeating: 0, count: dfOutRaw.count)
        for i in 0..<dfOutRaw.count {
            dfOut[i] = tanhf(dfOutRaw[i])
        }

        // Reshape dfOut to [nbDf, dfOrder * 2] and add c0p
        // dfOut layout: [nbDf * dfOrder * 2] in freq-major from grouped linear
        // c0p layout: from conv output, transposed to [B,T,F,C] -> [nbDf * dfOrder * 2]
        // The MLX code does:
        //   dfOut = tanh(groupedLinear(c)).reshaped([1,1,nbDf,dfOrder*2])
        //   c0p = c0p.transposed(0,2,3,1)  // BCHW -> BTHWC = [B, T, F, C] -> reshaped implicitly
        //   result = dfOut + c0p  where c0p has been transposed to [B,T,F_df,C]
        // Both should be [nbDf * dfOrder * 2] flat after reshape.
        // c0p after conv is in CHW layout: [outC * nbDf].
        // We need to transpose it to [nbDf * outC] (freq-major) to match dfOut's layout.
        // outC = dfOrder * 2 for the convp output (typically).
        // Wait, let me check the shapes more carefully.

        // The dfConvp produces output channels = dfConvp2Shape[0] (outC of second conv)
        // and outFreq = nbDf (no stride change).
        // The MLX streaming code does:
        //   c0p = c0p[0..., 0..., (t-1)..<t, 0...].transposed(0,2,3,1)
        // So c0p goes from BCHW [1, outC, 1, nbDf] -> transposed [1, 1, nbDf, outC]
        // Then it's added to dfOut.reshaped([1, 1, nbDf, dfOrder*2])
        // So outC of the conv chain should equal dfOrder * 2.

        // c0p is in CHW layout [outC, nbDf]. Transpose to freq-major [nbDf, outC].
        let c0pOutC = dfConvp2Shape.isEmpty ? 0 : dfConvp2Shape[0]
        let c0pFreqMajor: [Float]
        if c0p.count == c0pOutC * nbDf {
            c0pFreqMajor = flattenFreqMajor(c0p, channels: c0pOutC, freqDim: nbDf)
        } else {
            c0pFreqMajor = [Float](repeating: 0, count: dfOut.count)
        }

        // Add c0p to dfOut
        var result = [Float](repeating: 0, count: dfOut.count)
        let addCount = min(dfOut.count, c0pFreqMajor.count)
        for i in 0..<addCount {
            result[i] = dfOut[i] + c0pFreqMajor[i]
        }
        for i in addCount..<dfOut.count {
            result[i] = dfOut[i]
        }

        return result
    }

    /// Apply the DF convp chain (two convs + BN + relu) to dfConvpHistory.
    /// Returns the last time frame of the output in CHW layout.
    private func applyDfConvp() -> [Float] {
        guard dfConvp1Shape.count == 4, dfConvp2Shape.count == 4 else {
            return [Float](repeating: 0, count: convCh * nbDf)
        }

        let conv1Out = dfConvp1Shape[0]
        let conv1In = dfConvp1Shape[1]
        let conv1KH = dfConvp1Shape[2]
        let conv1KW = dfConvp1Shape[3]
        let conv2Out = dfConvp2Shape[0]
        let conv2In = dfConvp2Shape[1]
        let conv2KH = dfConvp2Shape[2]
        let conv2KW = dfConvp2Shape[3]

        // dfConvpHistory: [dfPathwayKernelSizeT] frames of [convCh * nbDf] in CHW layout
        let numFrames = dfConvpHistory.count
        let inFreq = nbDf

        // First conv: apply to history with time kernel
        let timePad1 = conv1KH - 1
        let totalTime1 = numFrames + timePad1
        let freqPad1 = conv1KW / 2
        let paddedFreq1 = inFreq + 2 * freqPad1

        // Build padded input
        var padded1 = [Float](repeating: 0, count: totalTime1 * conv1In * paddedFreq1)
        for t in 0..<numFrames {
            let tP = t + timePad1
            let frame = dfConvpHistory[t]
            for c in 0..<conv1In {
                for f in 0..<inFreq {
                    let srcIdx = c * inFreq + f
                    if srcIdx < frame.count {
                        padded1[(tP * conv1In + c) * paddedFreq1 + f + freqPad1] = frame[srcIdx]
                    }
                }
            }
        }

        let outTime1 = totalTime1 - conv1KH + 1
        let outFreq1 = (paddedFreq1 - conv1KW) / 1 + 1  // fstride=1
        var conv1Output = [Float](repeating: 0, count: outTime1 * conv1Out * outFreq1)

        for ot in 0..<outTime1 {
            for oc in 0..<conv1Out {
                for of in 0..<outFreq1 {
                    var sum: Float = 0
                    for ic in 0..<conv1In {
                        for kh in 0..<conv1KH {
                            let tIn = ot + kh
                            for kw in 0..<conv1KW {
                                let wIdx = ((oc * conv1In + ic) * conv1KH + kh) * conv1KW + kw
                                let xIdx = (tIn * conv1In + ic) * paddedFreq1 + of + kw
                                sum += dfConvp1Weight[wIdx] * padded1[xIdx]
                            }
                        }
                    }
                    conv1Output[(ot * conv1Out + oc) * outFreq1 + of] = sum
                }
            }
        }

        // Second conv on conv1 output
        let timePad2 = conv2KH - 1
        let totalTime2 = outTime1 + timePad2
        let freqPad2 = conv2KW / 2
        let paddedFreq2 = outFreq1 + 2 * freqPad2

        var padded2 = [Float](repeating: 0, count: totalTime2 * conv2In * paddedFreq2)
        for t in 0..<outTime1 {
            let tP = t + timePad2
            for c in 0..<conv2In {
                for f in 0..<outFreq1 {
                    padded2[(tP * conv2In + c) * paddedFreq2 + f + freqPad2] = conv1Output[(t * conv2In + c) * outFreq1 + f]
                }
            }
        }

        let outTime2 = totalTime2 - conv2KH + 1
        let outFreq2 = (paddedFreq2 - conv2KW) / 1 + 1  // fstride=1
        var conv2Output = [Float](repeating: 0, count: outTime2 * conv2Out * outFreq2)

        for ot in 0..<outTime2 {
            for oc in 0..<conv2Out {
                for of in 0..<outFreq2 {
                    var sum: Float = 0
                    for ic in 0..<conv2In {
                        for kh in 0..<conv2KH {
                            let tIn = ot + kh
                            for kw in 0..<conv2KW {
                                let wIdx = ((oc * conv2In + ic) * conv2KH + kh) * conv2KW + kw
                                let xIdx = (tIn * conv2In + ic) * paddedFreq2 + of + kw
                                sum += dfConvp2Weight[wIdx] * padded2[xIdx]
                            }
                        }
                    }
                    conv2Output[(ot * conv2Out + oc) * outFreq2 + of] = sum
                }
            }
        }

        // Apply BN + ReLU to conv2 output
        for ot in 0..<outTime2 {
            for c in 0..<conv2Out {
                for f in 0..<outFreq2 {
                    let idx = (ot * conv2Out + c) * outFreq2 + f
                    let val = conv2Output[idx] * dfConvpBnScale[c] + dfConvpBnBias[c]
                    conv2Output[idx] = max(0, val)
                }
            }
        }

        // Take last time frame
        let lastT = outTime2 - 1
        let frameSize = conv2Out * outFreq2
        if lastT >= 0 {
            return Array(conv2Output[(lastT * frameSize)..<((lastT + 1) * frameSize)])
        }
        return [Float](repeating: 0, count: frameSize)
    }

    // MARK: - Deep Filter

    private func applyDeepFilter(spec: [Float], specMasked: [Float], dfCoefs: [Float], dfSpecHistory: [[Float]]) -> [Float] {
        // dfCoefs: [nbDf * dfOrder * 2] in layout [f0_ord0_re, f0_ord0_im, f0_ord1_re, ..., f(nbDf-1)_ord(dfOrder-1)_im]
        // Actually the layout from the MLX code is:
        //   dfOut.reshaped([1, 1, nbDf, dfOrder * 2]) + c0p
        // where c0p is [1, 1, nbDf, dfOrder*2] (freq-major after transpose).
        // So dfCoefs[f * (dfOrder*2) + k*2 + 0] = real coef for freq f, order k
        // and dfCoefs[f * (dfOrder*2) + k*2 + 1] = imag coef for freq f, order k

        var specEnhanced = specMasked

        guard dfCoefs.count == nbDf * dfOrder * 2 else {
            return specEnhanced
        }

        for f in 0..<nbDf {
            var outRe: Float = 0
            var outIm: Float = 0

            for k in 0..<dfOrder {
                guard k < dfSpecHistory.count else { continue }
                let specFrame = dfSpecHistory[k]
                let sr = specFrame[2 * f]
                let si = specFrame[2 * f + 1]
                let coefIdx = f * (dfOrder * 2) + k * 2
                guard coefIdx + 1 < dfCoefs.count else { continue }
                let cr = dfCoefs[coefIdx]
                let ci = dfCoefs[coefIdx + 1]

                outRe += sr * cr - si * ci
                outIm += sr * ci + si * cr
            }

            if encConcat {
                specEnhanced[2 * f] = outRe
                specEnhanced[2 * f + 1] = outIm
            } else {
                // DF applies to unmasked spec for low freqs, masked spec for high freqs
                specEnhanced[2 * f] = outRe
                specEnhanced[2 * f + 1] = outIm
            }
        }

        // High frequencies: keep specMasked values (already in specEnhanced for f >= nbDf)
        return specEnhanced
    }

    // MARK: - Synthesis (ISTFT single frame)

    private func synthesisFrame(_ specNorm: [Float]) -> [Float] {
        let halfN = fftSize / 2

        var realIn = [Float](repeating: 0, count: halfN)
        var imagIn = [Float](repeating: 0, count: halfN)

        // DC
        realIn[0] = (specNorm[0] + specNorm[2 * halfN]) * 0.5
        imagIn[0] = (specNorm[0] - specNorm[2 * halfN]) * 0.5

        // Bins 1..halfN-1
        let invScale: Float = 0.5
        for k in 1..<halfN {
            realIn[k] = specNorm[2 * k] * invScale
            imagIn[k] = specNorm[2 * k + 1] * invScale
        }

        var realOut = [Float](repeating: 0, count: halfN)
        var imagOut = [Float](repeating: 0, count: halfN)
        vDSP_DFT_Execute(ifftSetup!, realIn, imagIn, &realOut, &imagOut)

        // Interleave to get time-domain signal
        var timeDomain = [Float](repeating: 0, count: fftSize)
        for i in 0..<halfN {
            timeDomain[2 * i] = realOut[i]
            timeDomain[2 * i + 1] = imagOut[i]
        }

        // IDFT normalization: vDSP_DFT_zrop round-trip has factor 2 with our DC/Nyquist handling
        let idftNorm: Float = 0.5
        vDSP_vsmul(timeDomain, 1, [idftNorm], &timeDomain, 1, vDSP_Length(fftSize))

        // Apply window
        vDSP_vmul(timeDomain, 1, vorbisWindow, 1, &timeDomain, 1, vDSP_Length(fftSize))

        // Overlap-add
        var out = [Float](repeating: 0, count: hopSize)
        vDSP_vadd(timeDomain, 1, synthMem, 1, &out, 1, vDSP_Length(hopSize))

        // Update synthesis memory
        let synthMemCount = synthMem.count
        if synthMemCount > 0 {
            let xSecond = Array(timeDomain[hopSize..<fftSize])
            if synthMemCount > hopSize {
                let split = synthMemCount - hopSize
                var newSynth = [Float](repeating: 0, count: synthMemCount)
                for i in 0..<split {
                    newSynth[i] = synthMem[hopSize + i] + xSecond[i]
                }
                for i in split..<synthMemCount {
                    newSynth[i] = xSecond[i]
                }
                synthMem = newSynth
            } else {
                synthMem = Array(xSecond[0..<synthMemCount])
            }
        }

        return out
    }

    // MARK: - Weight Extraction Helpers

    private func extractFloats(_ arr: MLXArray) -> [Float] {
        arr.asType(.float32).asArray(Float.self)
    }

    private func extractConvLayer(_ model: DeepFilterNetModel, prefix: String, main: Int, pointwise: Int?, bn: Int, fstride: Int) -> CPUConvLayer {
        let mainKey = "\(prefix).\(main).weight"
        let mainW = model.weights[mainKey]!.asType(.float32)
        let shape = mainW.shape  // [O, I, kH, kW]
        let mainData = extractFloats(mainW)

        var pwData: [Float]? = nil
        var pwOut = shape[0]
        if let pw = pointwise {
            let pwKey = "\(prefix).\(pw).weight"
            let pwW = model.weights[pwKey]!.asType(.float32)
            pwData = extractFloats(pwW)
            pwOut = pwW.shape[0]
        }

        let (bnScale, bnBias) = extractBNAffine(model, prefix: "\(prefix).\(bn)")

        return CPUConvLayer(
            weight: mainData,
            outChannels: shape[0],
            inChannels: shape[1],
            kH: shape[2],
            kW: shape[3],
            fstride: fstride,
            pointwiseWeight: pwData,
            pointwiseOut: pwOut,
            bnScale: bnScale,
            bnBias: bnBias
        )
    }

    private func extractBNAffine(_ model: DeepFilterNetModel, prefix: String) -> ([Float], [Float]) {
        guard let gamma = model.weights["\(prefix).weight"],
              let beta = model.weights["\(prefix).bias"],
              let mean = model.weights["\(prefix).running_mean"],
              let variance = model.weights["\(prefix).running_var"]
        else {
            return ([], [])
        }

        let g = gamma.asType(.float32).asArray(Float.self)
        let b = beta.asType(.float32).asArray(Float.self)
        let m = mean.asType(.float32).asArray(Float.self)
        let v = variance.asType(.float32).asArray(Float.self)

        let c = g.count
        var scale = [Float](repeating: 0, count: c)
        var bias = [Float](repeating: 0, count: c)
        for i in 0..<c {
            let s = g[i] / sqrtf(v[i] + 1e-5)
            scale[i] = s
            bias[i] = b[i] - m[i] * s
        }
        return (scale, bias)
    }

    private func extractSqueezedGRU(_ model: DeepFilterNetModel, prefix: String) -> CPUSqueezedGRU {
        let linInW = model.weights["\(prefix).linear_in.0.weight"]!.asType(.float32)
        var linInGroups = 1, linInIPG = 0, linInOPG = 0
        var linInData = [Float]()
        extractGroupedLinearFromArray(linInW, groups: &linInGroups, ipg: &linInIPG, opg: &linInOPG, weight: &linInData)

        var layers = [CPUGRULayer]()
        var layerIdx = 0
        while model.weights["\(prefix).gru.weight_ih_l\(layerIdx)"] != nil {
            let wih = model.weights["\(prefix).gru.weight_ih_l\(layerIdx)"]!.asType(.float32)
            let whh = model.weights["\(prefix).gru.weight_hh_l\(layerIdx)"]!.asType(.float32)
            let bih = model.weights["\(prefix).gru.bias_ih_l\(layerIdx)"]!.asType(.float32)
            let bhh = model.weights["\(prefix).gru.bias_hh_l\(layerIdx)"]!.asType(.float32)

            let hiddenSize = wih.shape[0] / 3
            let inputSize = wih.shape[1]

            // wih is [3H, I], we need wihT = [I, 3H]
            let wihData = extractFloats(wih.transposed())  // [I, 3H]
            let whhData = extractFloats(whh.transposed())  // [H, 3H]

            layers.append(CPUGRULayer(
                wihT: wihData,
                whhT: whhData,
                bih: extractFloats(bih),
                bhh: extractFloats(bhh),
                inputSize: inputSize,
                hiddenSize: hiddenSize
            ))
            layerIdx += 1
        }

        var linOutData: [Float]? = nil
        var linOutGroups = 1, linOutIPG = 0, linOutOPG = 0
        if let linOutW = model.weights["\(prefix).linear_out.0.weight"] {
            var d = [Float]()
            extractGroupedLinearFromArray(linOutW.asType(.float32), groups: &linOutGroups, ipg: &linOutIPG, opg: &linOutOPG, weight: &d)
            linOutData = d
        }

        return CPUSqueezedGRU(
            linearInWeight: linInData,
            linearInGroups: linInGroups,
            linearInInputPerGroup: linInIPG,
            linearInOutputPerGroup: linInOPG,
            layers: layers,
            linearOutWeight: linOutData,
            linearOutGroups: linOutGroups,
            linearOutInputPerGroup: linOutIPG,
            linearOutOutputPerGroup: linOutOPG
        )
    }

    private func extractGroupedLinear(_ model: DeepFilterNetModel, key: String,
                                       groups: inout Int, ipg: inout Int, opg: inout Int,
                                       weight: inout [Float]) {
        guard let w = model.weights[key] else { return }
        extractGroupedLinearFromArray(w.asType(.float32), groups: &groups, ipg: &ipg, opg: &opg, weight: &weight)
    }

    private func extractGroupedLinearFromArray(_ w: MLXArray, groups: inout Int, ipg: inout Int, opg: inout Int, weight: inout [Float]) {
        let shape = w.shape
        if shape.count == 3 {
            groups = shape[0]
            ipg = shape[1]
            opg = shape[2]
        } else if shape.count == 2 {
            groups = 1
            ipg = shape[1]
            opg = shape[0]
        }
        weight = extractFloats(w)
    }

    private func extractConvWeight(_ model: DeepFilterNetModel, key: String) -> ([Float], [Int]) {
        guard let w = model.weights[key] else { return ([], []) }
        return (extractFloats(w.asType(.float32)), w.shape)
    }

    private func extractPathwayConv(_ model: DeepFilterNetModel, prefix: String) -> CPUDecoderConvBlock? {
        guard let w = model.weights["\(prefix).0.weight"] else { return nil }
        let (bnS, bnB) = extractBNAffine(model, prefix: "\(prefix).1")
        return CPUDecoderConvBlock(
            conv0Weight: extractFloats(w.asType(.float32)),
            conv0Out: w.shape[0], conv0In: w.shape[1], conv0KH: w.shape[2], conv0KW: w.shape[3],
            conv1Weight: nil, conv1Out: 0, conv1In: 0, conv1KH: 0, conv1KW: 0,
            bnScale: bnS, bnBias: bnB,
            isTranspose: false, fstride: 1
        )
    }

    private func extractRegularBlock(_ model: DeepFilterNetModel, prefix: String) -> CPUDecoderConvBlock? {
        guard let w0 = model.weights["\(prefix).0.weight"],
              let w1 = model.weights["\(prefix).1.weight"] else { return nil }
        let (bnS, bnB) = extractBNAffine(model, prefix: "\(prefix).2")
        return CPUDecoderConvBlock(
            conv0Weight: extractFloats(w0.asType(.float32)),
            conv0Out: w0.shape[0], conv0In: w0.shape[1], conv0KH: w0.shape[2], conv0KW: w0.shape[3],
            conv1Weight: extractFloats(w1.asType(.float32)),
            conv1Out: w1.shape[0], conv1In: w1.shape[1], conv1KH: w1.shape[2], conv1KW: w1.shape[3],
            bnScale: bnS, bnBias: bnB,
            isTranspose: false, fstride: 1
        )
    }

    private func extractTransposeBlock(_ model: DeepFilterNetModel, prefix: String, fstride: Int) -> CPUDecoderConvBlock? {
        guard let w0 = model.weights["\(prefix).0.weight"],
              let w1 = model.weights["\(prefix).1.weight"] else { return nil }
        let (bnS, bnB) = extractBNAffine(model, prefix: "\(prefix).2")
        return CPUDecoderConvBlock(
            conv0Weight: extractFloats(w0.asType(.float32)),
            conv0Out: w0.shape[0], conv0In: w0.shape[1], conv0KH: w0.shape[2], conv0KW: w0.shape[3],
            conv1Weight: extractFloats(w1.asType(.float32)),
            conv1Out: w1.shape[0], conv1In: w1.shape[1], conv1KH: w1.shape[2], conv1KW: w1.shape[3],
            bnScale: bnS, bnBias: bnB,
            isTranspose: true, fstride: fstride
        )
    }

    private func extractOutputConv(_ model: DeepFilterNetModel, prefix: String) -> CPUDecoderConvBlock? {
        guard let w = model.weights["\(prefix).0.weight"] else { return nil }
        let (bnS, bnB) = extractBNAffine(model, prefix: "\(prefix).1")
        return CPUDecoderConvBlock(
            conv0Weight: extractFloats(w.asType(.float32)),
            conv0Out: w.shape[0], conv0In: w.shape[1], conv0KH: w.shape[2], conv0KW: w.shape[3],
            conv1Weight: nil, conv1Out: 0, conv1In: 0, conv1KH: 0, conv1KW: 0,
            bnScale: bnS, bnBias: bnB,
            isTranspose: false, fstride: 1
        )
    }

    // MARK: - Utilities

    private func linspace(start: Float, end: Float, count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        return (0..<count).map { start + Float($0) * step }
    }
}
