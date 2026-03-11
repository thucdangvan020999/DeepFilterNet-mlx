import Foundation
import MLX
import MLXFast

public struct DeepFilterNetPerformanceConfig: Sendable {
    public var enableMetalFusedMaskMultiply: Bool
    public var enableMetalFusedErbInvMaskApply: Bool
    public var enableMetalFusedStreamingDeepFilter: Bool
    public var enableMetalFusedOfflineDeepFilter: Bool
    public var enableMetalFusedGRUGates: Bool
    public var enableAccelerateGRU: Bool
    public var preferCompiledGraphs: Bool
    public var ensureKernelContiguousInputs: Bool
    public var kernelThreadGroupSize: Int

    public init(
        enableMetalFusedMaskMultiply: Bool = true,
        enableMetalFusedErbInvMaskApply: Bool = true,
        enableMetalFusedStreamingDeepFilter: Bool = true,
        enableMetalFusedOfflineDeepFilter: Bool = true,
        enableMetalFusedGRUGates: Bool = true,
        enableAccelerateGRU: Bool = true,
        preferCompiledGraphs: Bool = false,
        ensureKernelContiguousInputs: Bool = true,
        kernelThreadGroupSize: Int = 256
    ) {
        self.enableMetalFusedMaskMultiply = enableMetalFusedMaskMultiply
        self.enableMetalFusedErbInvMaskApply = enableMetalFusedErbInvMaskApply
        self.enableMetalFusedStreamingDeepFilter = enableMetalFusedStreamingDeepFilter
        self.enableMetalFusedOfflineDeepFilter = enableMetalFusedOfflineDeepFilter
        self.enableMetalFusedGRUGates = enableMetalFusedGRUGates
        self.enableAccelerateGRU = enableAccelerateGRU
        self.preferCompiledGraphs = preferCompiledGraphs
        self.ensureKernelContiguousInputs = ensureKernelContiguousInputs
        self.kernelThreadGroupSize = max(32, kernelThreadGroupSize)
    }

    public static let throughput = DeepFilterNetPerformanceConfig(
        enableMetalFusedMaskMultiply: true,
        enableMetalFusedErbInvMaskApply: true,
        enableMetalFusedStreamingDeepFilter: true,
        enableMetalFusedOfflineDeepFilter: true,
        enableMetalFusedGRUGates: true,
        enableAccelerateGRU: true,
        preferCompiledGraphs: true,
        ensureKernelContiguousInputs: true,
        kernelThreadGroupSize: 256
    )

    public static let safe = DeepFilterNetPerformanceConfig(
        enableMetalFusedMaskMultiply: false,
        enableMetalFusedErbInvMaskApply: false,
        enableMetalFusedStreamingDeepFilter: false,
        enableMetalFusedOfflineDeepFilter: false,
        enableMetalFusedGRUGates: false,
        enableAccelerateGRU: false,
        preferCompiledGraphs: false,
        ensureKernelContiguousInputs: true,
        kernelThreadGroupSize: 128
    )
}

final class DeepFilterNetKernelFusion: @unchecked Sendable {
    static let shared = DeepFilterNetKernelFusion()

    private let maskMultiplyKernelF32: MLXFast.MLXFastKernel
    private let maskMultiplyKernelF16: MLXFast.MLXFastKernel
    private let maskErbInvApplyKernelF32: MLXFast.MLXFastKernel
    private let maskErbInvApplyKernelF16: MLXFast.MLXFastKernel
    private let deepFilterFrameKernelF32: MLXFast.MLXFastKernel
    private let deepFilterFrameKernelF16: MLXFast.MLXFastKernel
    private let deepFilterFramePackedKernelF32: MLXFast.MLXFastKernel
    private let deepFilterFramePackedKernelF16: MLXFast.MLXFastKernel
    private let deepFilterOfflineKernelF32: MLXFast.MLXFastKernel
    private let deepFilterOfflineKernelF16: MLXFast.MLXFastKernel
    private let gruGateKernelF32: MLXFast.MLXFastKernel
    private let gruGateKernelF16: MLXFast.MLXFastKernel

    private init() {
        maskMultiplyKernelF32 = MLXFast.metalKernel(
            name: "dfn_mask_multiply_f32",
            inputNames: ["spec", "gains"],
            outputNames: ["out"],
            source: """
                uint idx = thread_position_in_grid.x;
                uint total = spec_shape[0] * spec_shape[1] * spec_shape[2] * spec_shape[3] * spec_shape[4];
                if (idx >= total) return;

                uint gainIdx = idx >> 1;
                out[idx] = spec[idx] * gains[gainIdx];
            """,
            ensureRowContiguous: true
        )

        maskMultiplyKernelF16 = MLXFast.metalKernel(
            name: "dfn_mask_multiply_f16",
            inputNames: ["spec", "gains"],
            outputNames: ["out"],
            source: """
                uint idx = thread_position_in_grid.x;
                uint total = spec_shape[0] * spec_shape[1] * spec_shape[2] * spec_shape[3] * spec_shape[4];
                if (idx >= total) return;

                uint gainIdx = idx >> 1;
                out[idx] = spec[idx] * gains[gainIdx];
            """,
            ensureRowContiguous: true
        )

        maskErbInvApplyKernelF32 = MLXFast.metalKernel(
            name: "dfn_mask_erbinv_apply_f32",
            inputNames: ["spec", "mask", "erbInvFB"],
            outputNames: ["out"],
            source: """
                uint idx = thread_position_in_grid.x;
                uint B = spec_shape[0];
                uint T = spec_shape[2];
                uint F = spec_shape[3];
                uint C = spec_shape[4];
                if (C != 2) return;
                uint total = B * T * F * C;
                if (idx >= total) return;

                uint i = idx;
                uint c = i % C; i /= C;
                uint f = i % F; i /= F;
                uint t = i % T; i /= T;
                uint b = i;

                uint E = mask_shape[3];
                float gain = 0.0f;
                for (uint e = 0; e < E; ++e) {
                    uint maskIdx = (((b * T + t) * E) + e);
                    uint fbIdx = e * F + f;
                    gain += mask[maskIdx] * erbInvFB[fbIdx];
                }

                uint specIdx = ((((b * T + t) * F + f) * C) + c);
                out[specIdx] = spec[specIdx] * gain;
            """,
            ensureRowContiguous: true
        )

        maskErbInvApplyKernelF16 = MLXFast.metalKernel(
            name: "dfn_mask_erbinv_apply_f16",
            inputNames: ["spec", "mask", "erbInvFB"],
            outputNames: ["out"],
            source: """
                uint idx = thread_position_in_grid.x;
                uint B = spec_shape[0];
                uint T = spec_shape[2];
                uint F = spec_shape[3];
                uint C = spec_shape[4];
                if (C != 2) return;
                uint total = B * T * F * C;
                if (idx >= total) return;

                uint i = idx;
                uint c = i % C; i /= C;
                uint f = i % F; i /= F;
                uint t = i % T; i /= T;
                uint b = i;

                uint E = mask_shape[3];
                float gain = 0.0f;
                for (uint e = 0; e < E; ++e) {
                    uint maskIdx = (((b * T + t) * E) + e);
                    uint fbIdx = e * F + f;
                    gain += (float)mask[maskIdx] * (float)erbInvFB[fbIdx];
                }

                uint specIdx = ((((b * T + t) * F + f) * C) + c);
                out[specIdx] = (float)spec[specIdx] * gain;
            """,
            ensureRowContiguous: true
        )

        deepFilterFrameKernelF32 = MLXFast.metalKernel(
            name: "dfn_stream_df_frame_f32",
            inputNames: ["specLow", "coefs"],
            outputNames: ["out"],
            source: """
                uint f = thread_position_in_grid.x;
                uint c = thread_position_in_grid.y;

                uint order = specLow_shape[0];
                uint bins = specLow_shape[1];
                if (f >= bins || c > 1) return;

                float outR = 0.0f;
                float outI = 0.0f;

                for (uint k = 0; k < order; ++k) {
                    uint base = (k * bins + f) * 2;
                    float sr = specLow[base + 0];
                    float si = specLow[base + 1];
                    float cr = coefs[base + 0];
                    float ci = coefs[base + 1];
                    outR += sr * cr - si * ci;
                    outI += sr * ci + si * cr;
                }

                uint outBase = f * 2;
                out[outBase + 0] = outR;
                out[outBase + 1] = outI;
            """,
            ensureRowContiguous: true
        )

        deepFilterFrameKernelF16 = MLXFast.metalKernel(
            name: "dfn_stream_df_frame_f16",
            inputNames: ["specLow", "coefs"],
            outputNames: ["out"],
            source: """
                uint f = thread_position_in_grid.x;
                uint c = thread_position_in_grid.y;

                uint order = specLow_shape[0];
                uint bins = specLow_shape[1];
                if (f >= bins || c > 1) return;

                float outR = 0.0f;
                float outI = 0.0f;

                for (uint k = 0; k < order; ++k) {
                    uint base = (k * bins + f) * 2;
                    float sr = (float)specLow[base + 0];
                    float si = (float)specLow[base + 1];
                    float cr = (float)coefs[base + 0];
                    float ci = (float)coefs[base + 1];
                    outR += sr * cr - si * ci;
                    outI += sr * ci + si * cr;
                }

                uint outBase = f * 2;
                out[outBase + 0] = outR;
                out[outBase + 1] = outI;
            """,
            ensureRowContiguous: true
        )

        deepFilterFramePackedKernelF32 = MLXFast.metalKernel(
            name: "dfn_stream_df_frame_packed_f32",
            inputNames: ["specLow", "coefsPacked"],
            outputNames: ["out"],
            source: """
                uint f = thread_position_in_grid.x;
                uint order = specLow_shape[0];
                uint bins = specLow_shape[1];
                if (f >= bins) return;

                float outR = 0.0f;
                float outI = 0.0f;
                uint coefBase = f * (order * 2);

                for (uint k = 0; k < order; ++k) {
                    uint sBase = (k * bins + f) * 2;
                    uint cBase = coefBase + (k * 2);
                    float sr = specLow[sBase + 0];
                    float si = specLow[sBase + 1];
                    float cr = coefsPacked[cBase + 0];
                    float ci = coefsPacked[cBase + 1];
                    outR += sr * cr - si * ci;
                    outI += sr * ci + si * cr;
                }

                uint outBase = f * 2;
                out[outBase + 0] = outR;
                out[outBase + 1] = outI;
            """,
            ensureRowContiguous: true
        )

        deepFilterFramePackedKernelF16 = MLXFast.metalKernel(
            name: "dfn_stream_df_frame_packed_f16",
            inputNames: ["specLow", "coefsPacked"],
            outputNames: ["out"],
            source: """
                uint f = thread_position_in_grid.x;
                uint order = specLow_shape[0];
                uint bins = specLow_shape[1];
                if (f >= bins) return;

                float outR = 0.0f;
                float outI = 0.0f;
                uint coefBase = f * (order * 2);

                for (uint k = 0; k < order; ++k) {
                    uint sBase = (k * bins + f) * 2;
                    uint cBase = coefBase + (k * 2);
                    float sr = (float)specLow[sBase + 0];
                    float si = (float)specLow[sBase + 1];
                    float cr = (float)coefsPacked[cBase + 0];
                    float ci = (float)coefsPacked[cBase + 1];
                    outR += sr * cr - si * ci;
                    outI += sr * ci + si * cr;
                }

                uint outBase = f * 2;
                out[outBase + 0] = outR;
                out[outBase + 1] = outI;
            """,
            ensureRowContiguous: true
        )

        deepFilterOfflineKernelF32 = MLXFast.metalKernel(
            name: "dfn_offline_df_f32",
            inputNames: ["specLow", "coefs", "padLeft"],
            outputNames: ["out"],
            source: """
                uint f = thread_position_in_grid.x;
                uint t = thread_position_in_grid.y;
                uint b = thread_position_in_grid.z;

                uint B = specLow_shape[0];
                uint T = specLow_shape[1];
                uint F = specLow_shape[2];
                uint K = coefs_shape[1];
                if (b >= B || t >= T || f >= F) return;

                int pad_left = (int)padLeft;
                float outR = 0.0f;
                float outI = 0.0f;

                for (uint k = 0; k < K; ++k) {
                    int srcT = int(t) + int(k) - pad_left;
                    if (srcT < 0 || srcT >= int(T)) continue;

                    uint sBase = (((b * T + uint(srcT)) * F + f) * 2);
                    uint cBase = ((((b * K + k) * T + t) * F + f) * 2);

                    float sr = specLow[sBase + 0];
                    float si = specLow[sBase + 1];
                    float cr = coefs[cBase + 0];
                    float ci = coefs[cBase + 1];

                    outR += sr * cr - si * ci;
                    outI += sr * ci + si * cr;
                }

                uint oBase = (((b * T + t) * F + f) * 2);
                out[oBase + 0] = outR;
                out[oBase + 1] = outI;
            """,
            ensureRowContiguous: true
        )

        deepFilterOfflineKernelF16 = MLXFast.metalKernel(
            name: "dfn_offline_df_f16",
            inputNames: ["specLow", "coefs", "padLeft"],
            outputNames: ["out"],
            source: """
                uint f = thread_position_in_grid.x;
                uint t = thread_position_in_grid.y;
                uint b = thread_position_in_grid.z;

                uint B = specLow_shape[0];
                uint T = specLow_shape[1];
                uint F = specLow_shape[2];
                uint K = coefs_shape[1];
                if (b >= B || t >= T || f >= F) return;

                int pad_left = (int)padLeft;
                float outR = 0.0f;
                float outI = 0.0f;

                for (uint k = 0; k < K; ++k) {
                    int srcT = int(t) + int(k) - pad_left;
                    if (srcT < 0 || srcT >= int(T)) continue;

                    uint sBase = (((b * T + uint(srcT)) * F + f) * 2);
                    uint cBase = ((((b * K + k) * T + t) * F + f) * 2);

                    float sr = (float)specLow[sBase + 0];
                    float si = (float)specLow[sBase + 1];
                    float cr = (float)coefs[cBase + 0];
                    float ci = (float)coefs[cBase + 1];

                    outR += sr * cr - si * ci;
                    outI += sr * ci + si * cr;
                }

                uint oBase = (((b * T + t) * F + f) * 2);
                out[oBase + 0] = outR;
                out[oBase + 1] = outI;
            """,
            ensureRowContiguous: true
        )

        gruGateKernelF32 = MLXFast.metalKernel(
            name: "dfn_gru_gate_step_f32",
            inputNames: ["gx", "gh", "prev"],
            outputNames: ["out"],
            source: """
                uint idx = thread_position_in_grid.x;
                uint bsz = prev_shape[0];
                uint hidden = prev_shape[1];
                uint total = bsz * hidden;
                if (idx >= total) return;

                uint b = idx / hidden;
                uint h = idx - b * hidden;
                uint base3 = b * hidden * 3;

                float xr = gx[base3 + h];
                float xz = gx[base3 + hidden + h];
                float xn = gx[base3 + 2 * hidden + h];
                float hr = gh[base3 + h];
                float hz = gh[base3 + hidden + h];
                float hn = gh[base3 + 2 * hidden + h];
                float prevh = prev[idx];

                float r = 1.0f / (1.0f + exp(-(xr + hr)));
                float z = 1.0f / (1.0f + exp(-(xz + hz)));
                float n = tanh(xn + r * hn);
                out[idx] = (1.0f - z) * n + z * prevh;
            """,
            ensureRowContiguous: true
        )

        gruGateKernelF16 = MLXFast.metalKernel(
            name: "dfn_gru_gate_step_f16",
            inputNames: ["gx", "gh", "prev"],
            outputNames: ["out"],
            source: """
                uint idx = thread_position_in_grid.x;
                uint bsz = prev_shape[0];
                uint hidden = prev_shape[1];
                uint total = bsz * hidden;
                if (idx >= total) return;

                uint b = idx / hidden;
                uint h = idx - b * hidden;
                uint base3 = b * hidden * 3;

                float xr = (float)gx[base3 + h];
                float xz = (float)gx[base3 + hidden + h];
                float xn = (float)gx[base3 + 2 * hidden + h];
                float hr = (float)gh[base3 + h];
                float hz = (float)gh[base3 + hidden + h];
                float hn = (float)gh[base3 + 2 * hidden + h];
                float prevh = (float)prev[idx];

                float r = 1.0f / (1.0f + exp(-(xr + hr)));
                float z = 1.0f / (1.0f + exp(-(xz + hz)));
                float n = tanh(xn + r * hn);
                out[idx] = (1.0f - z) * n + z * prevh;
            """,
            ensureRowContiguous: true
        )
    }

    func applyMaskMultiply(
        spec: MLXArray,
        gains: MLXArray,
        threadGroupSize: Int,
        ensureContiguous: Bool
    ) -> MLXArray? {
        guard spec.dtype == gains.dtype else {
            return nil
        }
        guard spec.dtype == .float32 || spec.dtype == .float16 else {
            return nil
        }
        guard spec.shape.count == 5, gains.shape.count == 5 else {
            return nil
        }
        guard spec.shape[4] == 2, gains.shape[4] == 1 else {
            return nil
        }

        let kernel = ensureContiguous
            ? (spec.dtype == .float16 ? maskMultiplyKernelF16 : maskMultiplyKernelF32)
            : MLXFast.metalKernel(
                name: spec.dtype == .float16 ? "dfn_mask_multiply_f16_nc" : "dfn_mask_multiply_f32_nc",
                inputNames: ["spec", "gains"],
                outputNames: ["out"],
                source: """
                    uint idx = thread_position_in_grid.x;
                    uint total = spec_shape[0] * spec_shape[1] * spec_shape[2] * spec_shape[3] * spec_shape[4];
                    if (idx >= total) return;
                    uint gainIdx = idx >> 1;
                    out[idx] = spec[idx] * gains[gainIdx];
                """,
                ensureRowContiguous: false
            )

        let tg = max(32, min(threadGroupSize, 1024))
        return kernel(
            [spec, gains],
            grid: (spec.size, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [spec.shape],
            outputDTypes: [spec.dtype]
        )[0]
    }

    func applyMaskErbInv(
        spec: MLXArray,
        mask: MLXArray,
        erbInvFB: MLXArray,
        threadGroupSize: Int
    ) -> MLXArray? {
        guard spec.dtype == mask.dtype, mask.dtype == erbInvFB.dtype else {
            return nil
        }
        guard spec.dtype == .float32 || spec.dtype == .float16 else {
            return nil
        }
        guard spec.shape.count == 5, mask.shape.count == 4, erbInvFB.shape.count == 2 else {
            return nil
        }
        guard spec.shape[1] == 1, mask.shape[1] == 1, spec.shape[4] == 2 else {
            return nil
        }
        let b = spec.shape[0]
        let t = spec.shape[2]
        let f = spec.shape[3]
        let e = mask.shape[3]
        guard mask.shape[0] == b, mask.shape[2] == t else { return nil }
        guard erbInvFB.shape[0] == e, erbInvFB.shape[1] == f else { return nil }

        let tg = max(32, min(threadGroupSize, 1024))
        let kernel = spec.dtype == .float16 ? maskErbInvApplyKernelF16 : maskErbInvApplyKernelF32
        return kernel(
            [spec, mask, erbInvFB],
            grid: (b * t * f * 2, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [spec.shape],
            outputDTypes: [spec.dtype]
        )[0]
    }

    func deepFilterStreamingFrame(
        specLow: MLXArray,
        coefs: MLXArray,
        threadGroupSize: Int,
        ensureContiguous: Bool
    ) -> MLXArray? {
        guard specLow.dtype == coefs.dtype else {
            return nil
        }
        guard specLow.dtype == .float32 || specLow.dtype == .float16 else {
            return nil
        }
        guard specLow.shape == coefs.shape, specLow.shape.count == 3 else {
            return nil
        }
        guard specLow.shape[2] == 2 else {
            return nil
        }

        let bins = specLow.shape[1]
        let tg = max(16, min(threadGroupSize, 256))

        let kernel = ensureContiguous
            ? (specLow.dtype == .float16 ? deepFilterFrameKernelF16 : deepFilterFrameKernelF32)
            : MLXFast.metalKernel(
                name: specLow.dtype == .float16 ? "dfn_stream_df_frame_f16_nc" : "dfn_stream_df_frame_f32_nc",
                inputNames: ["specLow", "coefs"],
                outputNames: ["out"],
                source: """
                    uint f = thread_position_in_grid.x;
                    uint c = thread_position_in_grid.y;
                    uint order = specLow_shape[0];
                    uint bins = specLow_shape[1];
                    if (f >= bins || c > 1) return;

                    float outR = 0.0f;
                    float outI = 0.0f;
                    for (uint k = 0; k < order; ++k) {
                        uint base = (k * bins + f) * 2;
                        float sr = specLow[base + 0];
                        float si = specLow[base + 1];
                        float cr = coefs[base + 0];
                        float ci = coefs[base + 1];
                        outR += sr * cr - si * ci;
                        outI += sr * ci + si * cr;
                    }

                    uint outBase = f * 2;
                    out[outBase + 0] = outR;
                    out[outBase + 1] = outI;
                """,
                ensureRowContiguous: false
            )

        return kernel(
            [specLow, coefs],
            grid: (bins, 2, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[bins, 2]],
            outputDTypes: [specLow.dtype]
        )[0]
    }

    func deepFilterStreamingFramePacked(
        specLow: MLXArray,
        coefsPacked: MLXArray,
        threadGroupSize: Int
    ) -> MLXArray? {
        guard specLow.dtype == coefsPacked.dtype else {
            return nil
        }
        guard specLow.dtype == .float32 || specLow.dtype == .float16 else {
            return nil
        }
        guard specLow.shape.count == 3, coefsPacked.shape.count == 4 else {
            return nil
        }
        guard specLow.shape[2] == 2 else {
            return nil
        }
        let order = specLow.shape[0]
        let bins = specLow.shape[1]
        guard coefsPacked.shape[0] == 1, coefsPacked.shape[1] == 1 else {
            return nil
        }
        guard coefsPacked.shape[2] == bins, coefsPacked.shape[3] == order * 2 else {
            return nil
        }

        let tg = max(32, min(threadGroupSize, 256))
        let kernel = specLow.dtype == .float16 ? deepFilterFramePackedKernelF16 : deepFilterFramePackedKernelF32
        return kernel(
            [specLow, coefsPacked],
            grid: (bins, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[bins, 2]],
            outputDTypes: [specLow.dtype]
        )[0]
    }

    func deepFilterOffline(
        specLow: MLXArray,
        coefs: MLXArray,
        padLeft: Int,
        threadGroupSize: Int,
        ensureContiguous: Bool
    ) -> MLXArray? {
        guard specLow.dtype == coefs.dtype else {
            return nil
        }
        guard specLow.dtype == .float32 || specLow.dtype == .float16 else {
            return nil
        }
        guard specLow.shape.count == 4, coefs.shape.count == 5 else {
            return nil
        }
        guard specLow.shape[0] == coefs.shape[0], specLow.shape[1] == coefs.shape[2], specLow.shape[2] == coefs.shape[3] else {
            return nil
        }
        guard specLow.shape[3] == 2, coefs.shape[4] == 2 else {
            return nil
        }

        let b = specLow.shape[0]
        let t = specLow.shape[1]
        let f = specLow.shape[2]
        let pad = MLXArray(Int32(max(0, padLeft))).asType(specLow.dtype)

        let kernel = ensureContiguous
            ? (specLow.dtype == .float16 ? deepFilterOfflineKernelF16 : deepFilterOfflineKernelF32)
            : MLXFast.metalKernel(
                name: specLow.dtype == .float16 ? "dfn_offline_df_f16_nc" : "dfn_offline_df_f32_nc",
                inputNames: ["specLow", "coefs", "padLeft"],
                outputNames: ["out"],
                source: """
                    uint f = thread_position_in_grid.x;
                    uint t = thread_position_in_grid.y;
                    uint b = thread_position_in_grid.z;
                    uint B = specLow_shape[0];
                    uint T = specLow_shape[1];
                    uint F = specLow_shape[2];
                    uint K = coefs_shape[1];
                    if (b >= B || t >= T || f >= F) return;

                    int pad_left = (int)padLeft;
                    float outR = 0.0f;
                    float outI = 0.0f;
                    for (uint k = 0; k < K; ++k) {
                        int srcT = int(t) + int(k) - pad_left;
                        if (srcT < 0 || srcT >= int(T)) continue;
                        uint sBase = (((b * T + uint(srcT)) * F + f) * 2);
                        uint cBase = ((((b * K + k) * T + t) * F + f) * 2);
                        float sr = specLow[sBase + 0];
                        float si = specLow[sBase + 1];
                        float cr = coefs[cBase + 0];
                        float ci = coefs[cBase + 1];
                        outR += sr * cr - si * ci;
                        outI += sr * ci + si * cr;
                    }

                    uint oBase = (((b * T + t) * F + f) * 2);
                    out[oBase + 0] = outR;
                    out[oBase + 1] = outI;
                """,
                ensureRowContiguous: false
            )

        let tg = max(16, min(threadGroupSize, 256))
        return kernel(
            [specLow, coefs, pad],
            grid: (f, t, b),
            threadGroup: (tg, 1, 1),
            outputShapes: [[b, t, f, 2]],
            outputDTypes: [specLow.dtype]
        )[0]
    }

    func gruGateStep(
        gx: MLXArray,
        gh: MLXArray,
        prev: MLXArray,
        hiddenSize: Int,
        threadGroupSize: Int
    ) -> MLXArray? {
        guard gx.dtype == gh.dtype, gh.dtype == prev.dtype else {
            return nil
        }
        guard gx.dtype == .float32 || gx.dtype == .float16 else {
            return nil
        }
        guard gx.ndim == 2, gh.ndim == 2, prev.ndim == 2 else {
            return nil
        }
        let b = prev.shape[0]
        let h = prev.shape[1]
        guard h == hiddenSize, gx.shape == [b, 3 * h], gh.shape == [b, 3 * h] else {
            return nil
        }

        let total = b * h
        let tg = max(32, min(threadGroupSize, 1024))
        let kernel = gx.dtype == .float16 ? gruGateKernelF16 : gruGateKernelF32
        return kernel(
            [gx, gh, prev],
            grid: (total, 1, 1),
            threadGroup: (tg, 1, 1),
            outputShapes: [[b, h]],
            outputDTypes: [gx.dtype]
        )[0]
    }
}
