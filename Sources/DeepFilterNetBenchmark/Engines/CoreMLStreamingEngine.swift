import Accelerate
import CoreML
import DeepFilterNetMLX
import Foundation
import MLX

// MARK: - CoreML Streaming Engine

/// CoreML + ANE streaming engine using the real DeepFilterNet3 CoreML model
/// from HuggingFace (`aufklarer/DeepFilterNet3-CoreML`).
///
/// The CoreML model processes the full sequence at once (internal GRU states
/// prevent per-hop streaming). Streaming is faked: `processHop` buffers all
/// frames and only produces output on `flush()`. Offline processing via
/// `enhanceOffline` is the natural mode.
public final class CoreMLStreamingEngine: StreamingEngine {
    public let engineType: StreamingEngineType = .coreML
    public var supportsOffline: Bool { true }

    // MARK: - CoreML model

    private var coreMLModel: MLModel?

    // MARK: - Config constants

    private var fftSize = 960
    private var hopSize = 480
    private var freqBins = 481   // fftSize / 2 + 1
    private var nbErb = 32
    private var nbDf = 96
    private var dfOrder = 5
    private var dfLookahead = 2
    private var convLookahead = 2
    private var normAlpha: Float = 0

    // MARK: - DSP state

    private var vorbisWindow = [Float]()
    private var analysisMem = [Float]()
    private var synthesisMem = [Float]()

    // ERB filterbank matrices (from the MLX model)
    private var erbFB = [Float]()       // [freqBins, nbErb] (forward)
    private var erbInvFB = [Float]()    // [nbErb, freqBins] (inverse)

    // Feature normalization running states
    private var meanNormState = [Float]()
    private var unitNormState = [Float]()
    private var meanNormStateInit = [Float]()
    private var unitNormStateInit = [Float]()

    // FFT setups (complex-to-complex, supports non-power-of-2 like 960)
    private var fftForwardSetup: OpaquePointer?
    private var fftInverseSetup: OpaquePointer?

    // MARK: - Streaming buffer

    /// Accumulated input audio for deferred processing
    private var inputBuffer = [Float]()

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
        normAlpha = model.normAlphaValue

        // Setup FFT (complex-to-complex via vDSP_DFT_zop)
        guard let fwd = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(fftSize), .FORWARD) else {
            throw CoreMLEngineError.fftSetupFailed
        }
        fftForwardSetup = fwd

        guard let inv = vDSP_DFT_zop_CreateSetup(nil, vDSP_Length(fftSize), .INVERSE) else {
            throw CoreMLEngineError.fftSetupFailed
        }
        fftInverseSetup = inv

        // Window and ERB filterbank from the MLX model
        vorbisWindow = model.vorbisWindow.asType(.float32).asArray(Float.self)

        // erbFB: the MLX model stores this as [freqBins, nbErb]
        erbFB = model.erbFB.asType(.float32).asArray(Float.self)
        // erbInvFB: [nbErb, freqBins]
        erbInvFB = model.erbInvFB.asType(.float32).asArray(Float.self)

        // Initialize normalization states
        initializeNormStates()

        // Try to load the CoreML model
        try loadCoreMLModel()

        // Initialize streaming state
        resetState()

        return CFAbsoluteTimeGetCurrent() - t0
    }

    // MARK: - CoreML Model Loading

    private static let coreMLRepoID = "aufklarer/DeepFilterNet3-CoreML"

    private func loadCoreMLModel() throws {
        let cacheDir = getCacheDirectory()

        let mlmodelcPath = cacheDir.appendingPathComponent("DeepFilterNet3.mlmodelc")
        let mlpackagePath = cacheDir.appendingPathComponent("DeepFilterNet3.mlpackage")

        if FileManager.default.fileExists(atPath: mlmodelcPath.path) {
            // Use pre-compiled model
            print("  [CoreML] Loading compiled model from: \(mlmodelcPath.path)")
            let config = MLModelConfiguration()
            config.computeUnits = .all
            coreMLModel = try MLModel(contentsOf: mlmodelcPath, configuration: config)
            loadAuxiliaryData(from: cacheDir)
            return
        }

        if FileManager.default.fileExists(atPath: mlpackagePath.path) {
            // Compile .mlpackage → .mlmodelc
            print("  [CoreML] Compiling model from: \(mlpackagePath.path)")
            let compiledURL = try MLModel.compileModel(at: mlpackagePath)

            // Copy compiled model to cache for future use
            let destURL = mlmodelcPath
            if !FileManager.default.fileExists(atPath: destURL.path) {
                try? FileManager.default.copyItem(at: compiledURL, to: destURL)
            }

            let config = MLModelConfiguration()
            config.computeUnits = .all
            coreMLModel = try MLModel(contentsOf: compiledURL, configuration: config)
            loadAuxiliaryData(from: cacheDir)
            return
        }

        // Model not found
        coreMLModel = nil
        print("  [CoreML] WARNING: No CoreML model found at: \(cacheDir.path)")
        print("  [CoreML] Download with: huggingface-cli download \(Self.coreMLRepoID)")
        print("  [CoreML] Expected files: DeepFilterNet3.mlpackage/, auxiliary.npz")
        print("  [CoreML] Will use passthrough (no enhancement) for benchmarking.")
    }

    // THOMAS EDITED CODE HERE 
    private func getCacheDirectory() -> URL {
        let fm = FileManager.default

        // 1. App's own Caches directory (iOS-safe)
        let appCaches = fm.urls(for: .cachesDirectory, in: .userDomainMask)[0]

        // Check HuggingFace-style subfolder inside app caches
        let hfCache = appCaches
            .appendingPathComponent("huggingface")
            .appendingPathComponent("hub")
            .appendingPathComponent("models--aufklarer--DeepFilterNet3-CoreML")
            .appendingPathComponent("snapshots")

        if let snapshots = try? fm.contentsOfDirectory(
            at: hfCache, includingPropertiesForKeys: nil
        ), let first = snapshots.first {
            return first
        }

        // 2. Custom cache directory inside app caches
        let customCache = appCaches
            .appendingPathComponent("DeepFilterNet3-CoreML")

        if fm.fileExists(atPath: customCache.path) {
            return customCache
        }

        // 3. DeepFilterNet subfolder inside app caches
        let deepfilterCache = appCaches
            .appendingPathComponent("deepfilternet")
            .appendingPathComponent("DeepFilterNet3-CoreML")

        if fm.fileExists(atPath: deepfilterCache.path) {
            return deepfilterCache
        }

        // Fallback: return custom cache path (created on demand)
        return customCache
    }

    /// Load auxiliary data (normalization states) from auxiliary.npz if present.
    /// The ERB filterbank and window are already loaded from the MLX model,
    /// but the npz may contain better initial normalization states.
    private func loadAuxiliaryData(from directory: URL) {
        let auxURL = directory.appendingPathComponent("auxiliary.npz")
        guard FileManager.default.fileExists(atPath: auxURL.path) else { return }

        do {
            let arrays = try NpzReader.read(url: auxURL)
            if let ms = arrays["mean_norm_state"], ms.count == nbErb {
                meanNormState = ms
                meanNormStateInit = ms
                print("  [CoreML] Loaded mean_norm_state from auxiliary.npz")
            }
            if let us = arrays["unit_norm_state"], us.count == nbDf {
                unitNormState = us
                unitNormStateInit = us
                print("  [CoreML] Loaded unit_norm_state from auxiliary.npz")
            }
            // Optionally override ERB filterbank from npz
            if let fb = arrays["erb_fb"], fb.count == freqBins * nbErb {
                erbFB = fb
                print("  [CoreML] Loaded erb_fb from auxiliary.npz")
            }
            if let ifb = arrays["erb_inv_fb"], ifb.count == nbErb * freqBins {
                erbInvFB = ifb
                print("  [CoreML] Loaded erb_inv_fb from auxiliary.npz")
            }
        } catch {
            print("  [CoreML] Warning: Failed to load auxiliary.npz: \(error)")
        }
    }

    // MARK: - Normalization State Initialization

    private func initializeNormStates() {
        // Default initialization matching the reference implementation
        // meanNormState: starts at low dB values (matching ERB dB range)
        meanNormState = linspace(start: -60.0, end: -90.0, count: nbErb)
        meanNormStateInit = meanNormState

        // unitNormState: starts at small magnitude values
        unitNormState = linspace(start: 0.001, end: 0.0001, count: nbDf)
        unitNormStateInit = unitNormState
    }

    // MARK: - State Management

    private func resetState() {
        let overlapSize = fftSize - hopSize
        analysisMem = [Float](repeating: 0, count: overlapSize)
        synthesisMem = [Float](repeating: 0, count: overlapSize)
        meanNormState = meanNormStateInit
        unitNormState = unitNormStateInit
        inputBuffer.removeAll()
    }

    public func reset() {
        resetState()
    }

    // MARK: - Streaming (Deferred)

    public func processHop(_ samples: [Float]) throws -> [Float] {
        // Buffer all input audio. The CoreML model processes the entire
        // sequence at once, so we defer actual processing to flush().
        inputBuffer.append(contentsOf: samples)
        return []
    }

    public func flush() throws -> [Float] {
        guard !inputBuffer.isEmpty else { return [] }
        guard coreMLModel != nil else {
            // No model: return silence
            let result = [Float](repeating: 0, count: inputBuffer.count)
            inputBuffer.removeAll()
            return result
        }

        let audio = inputBuffer
        inputBuffer.removeAll()

        return try processFullAudio(audio)
    }

    // MARK: - Offline

    public func enhanceOffline(_ audio: MLXArray, model: DeepFilterNetModel) throws -> MLXArray {
        guard coreMLModel != nil else {
            throw CoreMLEngineError.modelNotLoaded
        }

        let samples = audio.asType(.float32).asArray(Float.self)
        resetState()
        let enhanced = try processFullAudio(samples)
        return MLXArray(enhanced)
    }

    // MARK: - Full Audio Processing Pipeline

    /// Process the entire audio through the CoreML DeepFilterNet3 pipeline.
    ///
    /// Pipeline:
    /// 1. STFT analysis (Vorbis window, 960-point DFT)
    /// 2. Feature extraction (ERB power → dB → normalized, complex spec → unit normalized)
    /// 3. CoreML inference (produces ERB mask + DF coefficients)
    /// 4. Apply ERB mask to full spectrum
    /// 5. Apply deep filtering to lowest 96 bins
    /// 6. Combine: DF-enhanced for bins 0..<96, ERB-masked for bins 96..<481
    /// 7. ISTFT synthesis with overlap-add
    private func processFullAudio(_ audio: [Float]) throws -> [Float] {
        guard let model = coreMLModel else {
            throw CoreMLEngineError.modelNotLoaded
        }

        // Pad audio: one hop at start for analysis memory, one hop at end
        let paddedAudio = audio + [Float](repeating: 0, count: hopSize)

        // 1. STFT
        let (specReal, specImag) = stftForward(audio: paddedAudio)
        let numFrames = specReal.count / freqBins
        guard numFrames > 0 else { return [] }

        // 2. Compute ERB features [numFrames * nbErb]
        var erbFeats = computeERBFeatures(real: specReal, imag: specImag, numFrames: numFrames)

        // 3. Apply mean normalization to ERB features
        applyMeanNormalization(&erbFeats, numFrames: numFrames)

        // 4. Extract low-frequency spec for DF features
        var specFeatReal = [Float](repeating: 0, count: numFrames * nbDf)
        var specFeatImag = [Float](repeating: 0, count: numFrames * nbDf)
        for t in 0..<numFrames {
            for f in 0..<nbDf {
                specFeatReal[t * nbDf + f] = specReal[t * freqBins + f]
                specFeatImag[t * nbDf + f] = specImag[t * freqBins + f]
            }
        }

        // 5. Apply unit normalization to spec features
        applyUnitNormalization(real: &specFeatReal, imag: &specFeatImag, numFrames: numFrames)

        // 6. Build CoreML inputs
        // feat_erb: [1, 1, T, 32] (NCHW)
        let erbInput = try MLMultiArray(
            shape: [1, 1, numFrames as NSNumber, nbErb as NSNumber],
            dataType: .float32)
        let erbPtr = erbInput.dataPointer.assumingMemoryBound(to: Float.self)
        erbFeats.withUnsafeBufferPointer { src in
            erbPtr.update(from: src.baseAddress!, count: numFrames * nbErb)
        }

        // feat_spec: [1, 2, T, 96] (NCHW: channel 0=real, channel 1=imag)
        let specInput = try MLMultiArray(
            shape: [1, 2, numFrames as NSNumber, nbDf as NSNumber],
            dataType: .float32)
        let specPtr = specInput.dataPointer.assumingMemoryBound(to: Float.self)
        let channelStride = numFrames * nbDf
        specFeatReal.withUnsafeBufferPointer { src in
            specPtr.update(from: src.baseAddress!, count: channelStride)
        }
        specFeatImag.withUnsafeBufferPointer { src in
            (specPtr + channelStride).update(from: src.baseAddress!, count: channelStride)
        }

        // 7. CoreML inference
        let provider = try MLDictionaryFeatureProvider(dictionary: [
            "feat_erb": MLFeatureValue(multiArray: erbInput),
            "feat_spec": MLFeatureValue(multiArray: specInput),
        ])

        let prediction = try model.prediction(from: provider)

        guard let erbMaskArray = prediction.featureValue(for: "erb_mask")?.multiArrayValue,
              let coefsArray = prediction.featureValue(for: "df_coefs")?.multiArrayValue
        else {
            throw CoreMLEngineError.predictionOutputMissing
        }

        // 8. Extract ERB mask [1, 1, T, 32] → flat [T * 32]
        let erbMaskCount = numFrames * nbErb
        var erbMaskFlat = [Float](repeating: 0, count: erbMaskCount)
        extractMLMultiArrayFlat(erbMaskArray, into: &erbMaskFlat, count: erbMaskCount)

        // 9. Extract DF coefficients [1, 5, T, 96, 2]
        // Layout in memory (NCHW-ish): [O, T, F, 2] → reshape to [T, F, O, 2]
        let coefsCount = dfOrder * numFrames * nbDf * 2
        var coefsRaw = [Float](repeating: 0, count: coefsCount)
        extractMLMultiArrayFlat(coefsArray, into: &coefsRaw, count: coefsCount)

        // Reshape from CoreML layout [O, T, F, 2] to [T, F, O, 2]
        var coefsFlat = [Float](repeating: 0, count: coefsCount)
        for t in 0..<numFrames {
            for f in 0..<nbDf {
                for o in 0..<dfOrder {
                    let srcIdx = ((o * numFrames + t) * nbDf + f) * 2
                    let dstIdx = ((t * nbDf + f) * dfOrder + o) * 2
                    coefsFlat[dstIdx] = coefsRaw[srcIdx]
                    coefsFlat[dstIdx + 1] = coefsRaw[srcIdx + 1]
                }
            }
        }

        // 10. Apply ERB mask to full spectrum
        var enhancedReal = specReal
        var enhancedImag = specImag
        applyERBMask(
            specReal: &enhancedReal, specImag: &enhancedImag,
            erbMask: erbMaskFlat, numFrames: numFrames)

        // 11. Apply deep filtering to lowest nbDf bins
        let (dfReal, dfImag) = applyDeepFiltering(
            specReal: specReal, specImag: specImag,
            coefs: coefsFlat, numFrames: numFrames)

        // 12. Combine: DF-enhanced for bins 0..<nbDf, ERB-masked for rest
        for t in 0..<numFrames {
            for f in 0..<nbDf {
                enhancedReal[t * freqBins + f] = dfReal[t * nbDf + f]
                enhancedImag[t * freqBins + f] = dfImag[t * nbDf + f]
            }
        }

        // 13. ISTFT
        let rawOutput = stftInverse(real: enhancedReal, imag: enhancedImag, numFrames: numFrames)

        // 14. Trim: skip one hop of latency, take original audio length
        let trimStart = hopSize
        let trimEnd = min(trimStart + audio.count, rawOutput.count)
        guard trimEnd > trimStart else { return [] }
        return Array(rawOutput[trimStart..<trimEnd])
    }

    // MARK: - STFT Forward (Analysis)

    /// Analysis STFT: audio → (real, imag) arrays each of shape [numFrames * freqBins].
    /// Uses complex-to-complex DFT via vDSP_DFT_zop (supports non-power-of-2 sizes like 960).
    private func stftForward(audio: [Float]) -> (real: [Float], imag: [Float]) {
        let overlapSize = fftSize - hopSize
        let buffer = analysisMem + audio

        let numFrames = max(0, (buffer.count - fftSize) / hopSize + 1)
        guard numFrames > 0 else {
            analysisMem = Array(buffer.suffix(overlapSize))
            return ([], [])
        }

        var real = [Float](repeating: 0, count: numFrames * freqBins)
        var imag = [Float](repeating: 0, count: numFrames * freqBins)

        var windowedFrame = [Float](repeating: 0, count: fftSize)
        var zeroImag = [Float](repeating: 0, count: fftSize)
        var outReal = [Float](repeating: 0, count: fftSize)
        var outImag = [Float](repeating: 0, count: fftSize)

        for frame in 0..<numFrames {
            let start = frame * hopSize

            // Apply window
            buffer.withUnsafeBufferPointer { buf in
                vDSP_vmul(
                    buf.baseAddress! + start, 1,
                    vorbisWindow, 1,
                    &windowedFrame, 1,
                    vDSP_Length(fftSize))
            }

            // Zero imaginary input (real signal)
            vDSP_vclr(&zeroImag, 1, vDSP_Length(fftSize))

            // Complex DFT
            vDSP_DFT_Execute(
                fftForwardSetup!,
                windowedFrame, zeroImag,
                &outReal, &outImag)

            // Copy first freqBins (481) unique bins (conjugate symmetry)
            let baseIdx = frame * freqBins
            for k in 0..<freqBins {
                real[baseIdx + k] = outReal[k]
                imag[baseIdx + k] = outImag[k]
            }
        }

        // Update analysis memory
        let consumed = numFrames * hopSize
        analysisMem = Array(buffer.suffix(buffer.count - consumed))
        if analysisMem.count > overlapSize {
            analysisMem = Array(analysisMem.suffix(overlapSize))
        } else if analysisMem.count < overlapSize {
            analysisMem = [Float](repeating: 0, count: overlapSize - analysisMem.count) + analysisMem
        }

        return (real, imag)
    }

    // MARK: - STFT Inverse (Synthesis)

    /// Inverse STFT: complex spectrum → audio via overlap-add.
    private func stftInverse(real: [Float], imag: [Float], numFrames: Int) -> [Float] {
        guard numFrames > 0 else { return [] }

        let inverseScale: Float = 1.0 / Float(fftSize)
        let outputLen = numFrames * hopSize
        var output = [Float](repeating: 0, count: outputLen)

        var fullReal = [Float](repeating: 0, count: fftSize)
        var fullImag = [Float](repeating: 0, count: fftSize)
        var outReal = [Float](repeating: 0, count: fftSize)
        var outImag = [Float](repeating: 0, count: fftSize)

        for frame in 0..<numFrames {
            let baseIdx = frame * freqBins

            // Fill first freqBins (481) bins
            for k in 0..<freqBins {
                fullReal[k] = real[baseIdx + k]
                fullImag[k] = imag[baseIdx + k]
            }

            // Reconstruct conjugate symmetric part: X[N-k] = conj(X[k])
            for k in 1..<(fftSize / 2) {
                fullReal[fftSize - k] = fullReal[k]
                fullImag[fftSize - k] = -fullImag[k]
            }

            // Inverse DFT
            vDSP_DFT_Execute(
                fftInverseSetup!,
                fullReal, fullImag,
                &outReal, &outImag)

            // Scale by 1/N
            var scale = inverseScale
            vDSP_vsmul(outReal, 1, &scale, &outReal, 1, vDSP_Length(fftSize))

            // Apply synthesis window
            var windowed = [Float](repeating: 0, count: fftSize)
            vDSP_vmul(outReal, 1, vorbisWindow, 1, &windowed, 1, vDSP_Length(fftSize))

            // Overlap-add with synthesis memory
            let overlapCount = min(fftSize, synthesisMem.count)
            for i in 0..<overlapCount {
                windowed[i] += synthesisMem[i]
            }

            // Copy hop-size samples to output
            let outStart = frame * hopSize
            for i in 0..<hopSize {
                if outStart + i < outputLen {
                    output[outStart + i] = windowed[i]
                }
            }

            // Update synthesis memory
            let overlapSize = fftSize - hopSize
            synthesisMem = Array(windowed[hopSize..<fftSize])
            if synthesisMem.count < overlapSize {
                synthesisMem.append(
                    contentsOf: [Float](repeating: 0, count: overlapSize - synthesisMem.count))
            }
        }

        return output
    }

    // MARK: - Feature Extraction

    /// Compute ERB power features in dB from complex spectrum.
    /// Returns [numFrames * nbErb] in dB scale (not yet normalized).
    private func computeERBFeatures(
        real: [Float], imag: [Float], numFrames: Int
    ) -> [Float] {
        // Power spectrum: |X|^2
        var power = [Float](repeating: 0, count: numFrames * freqBins)
        for i in 0..<power.count {
            power[i] = real[i] * real[i] + imag[i] * imag[i]
        }

        // ERB compression: power[T, F] @ erbFB[F, B] → erb[T, B]
        var erb = [Float](repeating: 0, count: numFrames * nbErb)
        if !erbFB.isEmpty {
            vDSP_mmul(
                power, 1, erbFB, 1, &erb, 1,
                vDSP_Length(numFrames), vDSP_Length(nbErb), vDSP_Length(freqBins))
        }

        // Convert to dB: 10 * log10(erb + 1e-10)
        var count32 = Int32(erb.count)
        var epsilon: Float = 1e-10
        vDSP_vsadd(erb, 1, &epsilon, &erb, 1, vDSP_Length(erb.count))
        vvlog10f(&erb, erb, &count32)
        var scale: Float = 10.0
        vDSP_vsmul(erb, 1, &scale, &erb, 1, vDSP_Length(erb.count))

        return erb
    }

    /// Apply exponential mean normalization to ERB features.
    /// state = x * (1 - alpha) + state * alpha; x_norm = (x - state) / 40
    private func applyMeanNormalization(_ erb: inout [Float], numFrames: Int) {
        let oneMinusAlpha = 1.0 - normAlpha
        for t in 0..<numFrames {
            let baseIdx = t * nbErb
            for b in 0..<nbErb {
                let x = erb[baseIdx + b]
                meanNormState[b] = x * oneMinusAlpha + meanNormState[b] * normAlpha
                erb[baseIdx + b] = (x - meanNormState[b]) / 40.0
            }
        }
    }

    /// Apply exponential unit normalization to complex spec features.
    /// state = |x| * (1 - alpha) + state * alpha; x_norm = x / sqrt(state)
    private func applyUnitNormalization(
        real: inout [Float], imag: inout [Float], numFrames: Int
    ) {
        let oneMinusAlpha = 1.0 - normAlpha
        for t in 0..<numFrames {
            let baseIdx = t * nbDf
            for f in 0..<nbDf {
                let re = real[baseIdx + f]
                let im = imag[baseIdx + f]
                let mag = sqrtf(re * re + im * im)
                unitNormState[f] = mag * oneMinusAlpha + unitNormState[f] * normAlpha
                let norm = sqrtf(max(unitNormState[f], 1e-10))
                real[baseIdx + f] = re / norm
                imag[baseIdx + f] = im / norm
            }
        }
    }

    // MARK: - Post-processing

    /// Extract float32 data from an MLMultiArray, handling float16 output from CoreML.
    private func extractMLMultiArrayFlat(
        _ array: MLMultiArray, into output: inout [Float], count: Int
    ) {
        if array.dataType == .float16 {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float16.self)
            for i in 0..<count {
                output[i] = Float(ptr[i])
            }
        } else {
            let ptr = array.dataPointer.assumingMemoryBound(to: Float.self)
            output.withUnsafeMutableBufferPointer { dst in
                dst.baseAddress!.update(from: ptr, count: count)
            }
        }
    }

    /// Apply ERB mask to the full spectrum.
    /// Expands the 32-band mask to 481 bins using the inverse filterbank,
    /// then multiplies the spectrum by the per-frequency gain.
    private func applyERBMask(
        specReal: inout [Float], specImag: inout [Float],
        erbMask: [Float], numFrames: Int
    ) {
        // Expand mask: mask[T, B] @ invFb[B, F] → fullMask[T, F]
        var fullMask = [Float](repeating: 0, count: numFrames * freqBins)
        if !erbInvFB.isEmpty {
            vDSP_mmul(
                erbMask, 1, erbInvFB, 1, &fullMask, 1,
                vDSP_Length(numFrames), vDSP_Length(freqBins), vDSP_Length(nbErb))
        }

        // Apply mask to spectrum
        vDSP_vmul(specReal, 1, fullMask, 1, &specReal, 1, vDSP_Length(specReal.count))
        vDSP_vmul(specImag, 1, fullMask, 1, &specImag, 1, vDSP_Length(specImag.count))
    }

    /// Apply deep filtering coefficients to the spectrum.
    ///
    /// For each time t and frequency f (0..<nbDf):
    ///   Y(t, f) = sum_{n=0}^{order-1} X(t + n - padBefore, f) * W(n, t, f)
    /// where multiplication is complex.
    private func applyDeepFiltering(
        specReal: [Float], specImag: [Float],
        coefs: [Float], numFrames: Int
    ) -> (real: [Float], imag: [Float]) {
        let padBefore = dfOrder - 1 - dfLookahead  // 5 - 1 - 2 = 2

        var outReal = [Float](repeating: 0, count: numFrames * nbDf)
        var outImag = [Float](repeating: 0, count: numFrames * nbDf)

        for t in 0..<numFrames {
            for f in 0..<nbDf {
                var sumRe: Float = 0
                var sumIm: Float = 0

                for n in 0..<dfOrder {
                    let srcT = t + n - padBefore
                    let coefIdx = (t * nbDf * dfOrder + f * dfOrder + n) * 2

                    let wRe = coefs[coefIdx]
                    let wIm = coefs[coefIdx + 1]

                    // Clamp source frame index
                    let clampedT = max(0, min(numFrames - 1, srcT))
                    let srcIdx = clampedT * freqBins + f

                    let xRe = specReal[srcIdx]
                    let xIm = specImag[srcIdx]

                    // Complex multiply: (xRe + j*xIm) * (wRe + j*wIm)
                    sumRe += xRe * wRe - xIm * wIm
                    sumIm += xIm * wRe + xRe * wIm
                }

                let outIdx = t * nbDf + f
                outReal[outIdx] = sumRe
                outImag[outIdx] = sumIm
            }
        }

        return (outReal, outImag)
    }

    // MARK: - Utilities

    private func linspace(start: Float, end: Float, count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        return (0..<count).map { start + Float($0) * step }
    }
}

// MARK: - NPZ Reader

/// Minimal reader for NumPy .npz files (uncompressed, float32 only).
private enum NpzReader {

    static func read(url: URL) throws -> [String: [Float]] {
        let data = try Data(contentsOf: url)
        var result = [String: [Float]]()

        var offset = 0
        while offset + 30 <= data.count {
            // ZIP local file header signature
            let b0 = data[offset], b1 = data[offset + 1]
            let b2 = data[offset + 2], b3 = data[offset + 3]
            guard b0 == 0x50 && b1 == 0x4B && b2 == 0x03 && b3 == 0x04 else { break }

            var compressedSize = Int(readUInt32(data, at: offset + 18))
            var uncompressedSize = Int(readUInt32(data, at: offset + 22))
            let nameLen = Int(readUInt16(data, at: offset + 26))
            let extraLen = Int(readUInt16(data, at: offset + 28))

            let nameStart = offset + 30
            guard nameStart + nameLen <= data.count else { break }
            let nameData = data.subdata(in: nameStart..<nameStart + nameLen)
            var name = String(data: nameData, encoding: .utf8) ?? ""

            if name.hasSuffix(".npy") {
                name = String(name.dropLast(4))
            }

            // Handle ZIP64 extra field
            if compressedSize == 0xFFFF_FFFF || uncompressedSize == 0xFFFF_FFFF {
                let extraStart = nameStart + nameLen
                if extraLen >= 4 {
                    let tag = readUInt16(data, at: extraStart)
                    if tag == 0x0001 {
                        var extraOffset = extraStart + 4
                        if uncompressedSize == 0xFFFF_FFFF {
                            uncompressedSize = Int(readUInt64(data, at: extraOffset))
                            extraOffset += 8
                        }
                        if compressedSize == 0xFFFF_FFFF {
                            compressedSize = Int(readUInt64(data, at: extraOffset))
                        }
                    }
                }
            }

            let dataStart = nameStart + nameLen + extraLen
            let dataSize = max(compressedSize, uncompressedSize)
            guard dataStart + dataSize <= data.count else { break }

            if let floats = parseNpy(data, npyOffset: dataStart, npySize: uncompressedSize) {
                result[name] = floats
            }

            offset = dataStart + dataSize
        }

        return result
    }

    private static func parseNpy(_ data: Data, npyOffset: Int, npySize: Int) -> [Float]? {
        guard npySize >= 10,
              data[npyOffset] == 0x93,
              data[npyOffset + 1] == 0x4E
        else { return nil }

        let majorVersion = data[npyOffset + 6]

        let headerLen: Int
        if majorVersion == 1 {
            headerLen = Int(readUInt16(data, at: npyOffset + 8))
        } else {
            headerLen = Int(readUInt32(data, at: npyOffset + 8))
        }

        let headerSize = (majorVersion == 1) ? 10 : 12
        let floatStart = npyOffset + headerSize + headerLen
        let numBytes = npySize - headerSize - headerLen

        guard numBytes > 0 else { return nil }
        let numFloats = numBytes / 4

        var result = [Float](repeating: 0, count: numFloats)
        _ = result.withUnsafeMutableBytes { dst in
            data.copyBytes(to: dst, from: floatStart..<floatStart + numFloats * 4)
        }
        return result
    }

    private static func readUInt16(_ data: Data, at offset: Int) -> UInt16 {
        data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset - data.startIndex, as: UInt16.self)
        }
    }

    private static func readUInt32(_ data: Data, at offset: Int) -> UInt32 {
        data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset - data.startIndex, as: UInt32.self)
        }
    }

    private static func readUInt64(_ data: Data, at offset: Int) -> UInt64 {
        data.withUnsafeBytes { raw in
            raw.loadUnaligned(fromByteOffset: offset - data.startIndex, as: UInt64.self)
        }
    }
}

// MARK: - Error Types

enum CoreMLEngineError: Error, LocalizedError {
    case modelNotLoaded
    case predictionOutputMissing
    case fftSetupFailed

    var errorDescription: String? {
        switch self {
        case .modelNotLoaded:
            return "CoreML model not loaded. Download from: aufklarer/DeepFilterNet3-CoreML"
        case .predictionOutputMissing:
            return "CoreML prediction did not produce expected outputs (erb_mask, df_coefs)"
        case .fftSetupFailed:
            return "Failed to create vDSP DFT setup for FFT"
        }
    }
}
