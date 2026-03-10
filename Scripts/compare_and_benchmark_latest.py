#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import matplotlib.pyplot as plt
    import soundfile as sf
    from matplotlib.patches import Patch
except Exception as exc:  # pragma: no cover
    raise RuntimeError("matplotlib and soundfile are required") from exc


@dataclass
class RunSpec:
    name: str
    mode: str  # offline | stream
    cmd: List[str]
    raw_output: Path
    final_output: Path
    cwd: Optional[Path] = None
    env: Optional[Dict[str, str]] = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def copy_metallib_bundle(swift_repo: Path) -> None:
    source_candidates = [
        swift_repo / ".xcodebuild/Build/Products/Release/mlx-swift_Cmlx.bundle",
        swift_repo / ".xcodebuild/Build/Products/Debug/mlx-swift_Cmlx.bundle",
        swift_repo / ".build/arm64-apple-macosx/debug/mlx-swift_Cmlx.bundle",
    ]
    source = next((p for p in source_candidates if p.exists()), None)
    if source is None:
        return

    for config in ("debug", "release"):
        dest_parent = swift_repo / f".build/arm64-apple-macosx/{config}"
        dest = dest_parent / "mlx-swift_Cmlx.bundle"
        ensure_dir(dest_parent)
        if not dest.exists():
            shutil.copytree(source, dest)


def run_cmd(spec: RunSpec, repeats: int) -> Tuple[List[float], Optional[str]]:
    times: List[float] = []
    error: Optional[str] = None

    if spec.final_output.exists():
        spec.final_output.unlink()

    for _ in range(max(1, repeats)):
        t0 = time.perf_counter()
        proc = subprocess.run(
            spec.cmd,
            cwd=str(spec.cwd) if spec.cwd else None,
            env=spec.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        dt = time.perf_counter() - t0

        if proc.returncode != 0:
            error = (
                f"exit={proc.returncode}\n"
                f"command={' '.join(spec.cmd)}\n"
                f"output:\n{proc.stdout[-4000:]}"
            )
            break
        if not spec.raw_output.exists():
            error = (
                f"Output missing for {spec.name}: {spec.raw_output}\n"
                f"command={' '.join(spec.cmd)}\n"
                f"output:\n{proc.stdout[-4000:]}"
            )
            break
        times.append(dt)

    if error is None:
        if spec.raw_output != spec.final_output:
            shutil.copy2(spec.raw_output, spec.final_output)
        elif not spec.final_output.exists():
            error = f"Final output missing for {spec.name}: {spec.final_output}"

    return times, error


def load_mono(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def fft_best_lag(ref: np.ndarray, x: np.ndarray, max_lag: int) -> int:
    n = len(ref)
    m = len(x)
    size = 1 << int(math.ceil(math.log2(n + m - 1)))

    fr = np.fft.rfft(ref, size)
    fx = np.fft.rfft(x, size)
    corr_circular = np.fft.irfft(fr * np.conj(fx), size)

    corr_linear = np.concatenate((corr_circular[-(m - 1):], corr_circular[:n]))
    lags = np.arange(-(m - 1), n)

    mask = (lags >= -max_lag) & (lags <= max_lag)
    lags_w = lags[mask]
    corr_w = corr_linear[mask]

    return int(lags_w[int(np.argmax(corr_w))])


def align_signals(ref: np.ndarray, x: np.ndarray, max_lag: int = 4096) -> Tuple[np.ndarray, np.ndarray, int]:
    lag = fft_best_lag(ref, x, max_lag=max_lag)

    if lag > 0:
        ref_s = ref[lag:]
        x_s = x[: len(ref_s)]
    elif lag < 0:
        shift = -lag
        x_s = x[shift:]
        ref_s = ref[: len(x_s)]
    else:
        n = min(len(ref), len(x))
        ref_s = ref[:n]
        x_s = x[:n]

    n = min(len(ref_s), len(x_s))
    return ref_s[:n], x_s[:n], lag


def metrics_vs_ref(ref: np.ndarray, x: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    ref64 = ref.astype(np.float64)
    x64 = x.astype(np.float64)

    ref_c = ref64 - ref64.mean()
    x_c = x64 - x64.mean()
    pearson = float(np.dot(ref_c, x_c) / (np.linalg.norm(ref_c) * np.linalg.norm(x_c) + eps))
    cosine = float(np.dot(ref64, x64) / (np.linalg.norm(ref64) * np.linalg.norm(x64) + eps))

    err = x64 - ref64
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    snr_db = float(10.0 * np.log10((np.mean(ref64 * ref64) + eps) / (mse + eps)))
    return {
        "pearson": pearson,
        "cosine": cosine,
        "mse": mse,
        "mae": mae,
        "snr_db": snr_db,
    }


def stft_mag_db(audio: np.ndarray, n_fft: int = 960, hop: int = 480) -> np.ndarray:
    if len(audio) < n_fft:
        pad = np.zeros(n_fft - len(audio), dtype=np.float32)
        audio = np.concatenate([audio, pad], axis=0)
    window = np.hanning(n_fft).astype(np.float32)
    frames: List[np.ndarray] = []
    for start in range(0, len(audio) - n_fft + 1, hop):
        frame = audio[start : start + n_fft] * window
        spec = np.fft.rfft(frame)
        frames.append(20.0 * np.log10(np.abs(spec) + 1e-8))
    return np.stack(frames, axis=1) if frames else np.zeros((n_fft // 2 + 1, 1), dtype=np.float32)


def resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr:
        return audio
    if len(audio) == 0:
        return audio
    duration = len(audio) / float(src_sr)
    dst_len = max(1, int(round(duration * dst_sr)))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def trim3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = min(len(a), len(b), len(c))
    return a[:n], b[:n], c[:n]


def stft_with_axes(audio: np.ndarray, sr: int, n_fft: int = 960, hop: int = 480) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    spec_db = stft_mag_db(audio, n_fft=n_fft, hop=hop)
    f = np.fft.rfftfreq(n_fft, d=1.0 / float(sr))
    t = np.arange(spec_db.shape[1], dtype=np.float32) * (hop / float(sr))
    return f, t, spec_db


def compute_rms_energy(audio: np.ndarray, window_size: int = 4800, hop: int = 2400) -> np.ndarray:
    if len(audio) < window_size:
        audio = np.pad(audio, (0, window_size - len(audio)))
    n_windows = max(1, (len(audio) - window_size) // hop + 1)
    rms = np.zeros(n_windows, dtype=np.float32)
    for i in range(n_windows):
        start = i * hop
        window = audio[start : start + window_size]
        rms[i] = float(np.sqrt(np.mean(window * window) + 1e-12))
    return rms


def compute_detailed_metrics(original: np.ndarray, reference: np.ndarray, test: np.ndarray, sr: int) -> Dict[str, float]:
    orig, ref, x = trim3(original, reference, test)
    eps = 1e-12
    ref64 = ref.astype(np.float64)
    x64 = x.astype(np.float64)
    orig64 = orig.astype(np.float64)

    noise_ref = orig64 - ref64
    noise_test = orig64 - x64
    diff = ref64 - x64

    corr = float(np.corrcoef(ref64, x64)[0, 1]) if len(ref64) > 1 else 1.0
    mse = float(np.mean(diff * diff))
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(mse))
    max_abs = float(np.max(np.abs(diff)))
    signal_power = float(np.mean(ref64 * ref64))
    ser_db = float(10.0 * np.log10((signal_power + eps) / (mse + eps)))

    _, _, spec_orig = stft_with_axes(orig, sr)
    _, _, spec_ref = stft_with_axes(ref, sr)
    _, _, spec_test = stft_with_axes(x, sr)
    tmin = min(spec_orig.shape[1], spec_ref.shape[1], spec_test.shape[1])
    spec_orig = spec_orig[:, :tmin]
    spec_ref = spec_ref[:, :tmin]
    spec_test = spec_test[:, :tmin]
    spec_diff = spec_ref - spec_test

    reduction_ref = spec_orig - spec_ref
    reduction_test = spec_orig - spec_test
    reduction_diff = reduction_ref - reduction_test

    rms_orig = compute_rms_energy(orig)
    rms_ref = compute_rms_energy(ref)
    rms_test = compute_rms_energy(x)
    rmin = min(len(rms_orig), len(rms_ref), len(rms_test))

    return {
        "duration_sec": float(len(orig) / float(sr)),
        "rms_original": float(np.sqrt(np.mean(orig64 * orig64))),
        "rms_reference": float(np.sqrt(np.mean(ref64 * ref64))),
        "rms_test": float(np.sqrt(np.mean(x64 * x64))),
        "noise_rms_reference": float(np.sqrt(np.mean(noise_ref * noise_ref))),
        "noise_rms_test": float(np.sqrt(np.mean(noise_test * noise_test))),
        "correlation": corr,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "max_abs_diff": max_abs,
        "ser_db": ser_db,
        "spectral_mae_db": float(np.mean(np.abs(spec_diff))),
        "spectral_rmse_db": float(np.sqrt(np.mean(spec_diff * spec_diff))),
        "spectral_max_abs_db": float(np.max(np.abs(spec_diff))),
        "spectral_mae_db_active_gt_-70db": float(np.mean(np.abs(spec_diff)[spec_ref > -70.0])) if np.any(spec_ref > -70.0) else 0.0,
        "spectral_mae_db_active_gt_-60db": float(np.mean(np.abs(spec_diff)[spec_ref > -60.0])) if np.any(spec_ref > -60.0) else 0.0,
        "spectral_mae_db_active_gt_-50db": float(np.mean(np.abs(spec_diff)[spec_ref > -50.0])) if np.any(spec_ref > -50.0) else 0.0,
        "spectral_p95_abs_db": float(np.percentile(np.abs(spec_diff), 95.0)),
        "noise_reduction_diff_mae_db": float(np.mean(np.abs(reduction_diff))),
        "noise_reduction_diff_rmse_db": float(np.sqrt(np.mean(reduction_diff * reduction_diff))),
        "noise_reduction_diff_max_abs_db": float(np.max(np.abs(reduction_diff))),
        "rms_trace_mae": float(np.mean(np.abs(rms_ref[:rmin] - rms_test[:rmin]))),
    }


def plot_waveform_comparison(original: np.ndarray, reference: np.ndarray, test: np.ndarray, sr: int, output_path: Path, title_suffix: str) -> None:
    orig, ref, x = trim3(original, reference, test)
    fig, axes = plt.subplots(3, 1, figsize=(15, 8))
    t = np.arange(len(orig)) / float(sr)
    s1 = min(len(orig), sr)
    axes[0].plot(t[:s1], orig[:s1], "b-", alpha=0.7, label="Original")
    axes[0].plot(t[:s1], ref[:s1], "r-", alpha=0.5, label="Reference")
    axes[0].plot(t[:s1], x[:s1], "g-", alpha=0.5, label="Test")
    axes[0].set_title(f"Waveform Comparison (First 1s) - {title_suffix}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(t, orig, "b-", alpha=0.6, label="Original")
    axes[1].plot(t, ref, "r-", alpha=0.5, label="Reference")
    axes[1].plot(t, x, "g-", alpha=0.5, label="Test")
    axes[1].set_title(f"Waveform Comparison (Full) - {title_suffix}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Amplitude")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    diff = ref - x
    axes[2].plot(t, diff, color="purple", alpha=0.8)
    axes[2].axhline(y=0.0, color="k", linestyle="--", alpha=0.5)
    axes[2].set_title(f"Reference - Test Difference - {title_suffix}")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Difference")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_spectrogram_comparison(original: np.ndarray, reference: np.ndarray, test: np.ndarray, sr: int, output_path: Path, title_suffix: str) -> None:
    orig, ref, x = trim3(original, reference, test)
    f, t_orig, spec_orig = stft_with_axes(orig, sr)
    _, t_ref, spec_ref = stft_with_axes(ref, sr)
    _, t_test, spec_test = stft_with_axes(x, sr)
    tmin = min(spec_orig.shape[1], spec_ref.shape[1], spec_test.shape[1])
    t = t_orig[:tmin]
    spec_orig = spec_orig[:, :tmin]
    spec_ref = spec_ref[:, :tmin]
    spec_test = spec_test[:, :tmin]

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    vmin, vmax = -80, 0

    im0 = axes[0, 0].pcolormesh(t, f, spec_orig, shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0, 0].set_title("Original")
    axes[0, 0].set_xlabel("Time (s)")
    axes[0, 0].set_ylabel("Frequency (Hz)")
    plt.colorbar(im0, ax=axes[0, 0], label="dB")

    im1 = axes[0, 1].pcolormesh(t, f, spec_ref, shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0, 1].set_title("Reference")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Frequency (Hz)")
    plt.colorbar(im1, ax=axes[0, 1], label="dB")

    im2 = axes[0, 2].pcolormesh(t, f, spec_test, shading="gouraud", vmin=vmin, vmax=vmax, cmap="viridis")
    axes[0, 2].set_title("Test")
    axes[0, 2].set_xlabel("Time (s)")
    axes[0, 2].set_ylabel("Frequency (Hz)")
    plt.colorbar(im2, ax=axes[0, 2], label="dB")

    diff_db = spec_ref - spec_test
    im3 = axes[1, 0].pcolormesh(t, f, diff_db, shading="gouraud", vmin=-20, vmax=20, cmap="RdBu_r")
    axes[1, 0].set_title("Reference - Test (dB)")
    axes[1, 0].set_xlabel("Time (s)")
    axes[1, 0].set_ylabel("Frequency (Hz)")
    plt.colorbar(im3, ax=axes[1, 0], label="dB diff")

    reduction_ref = spec_orig - spec_ref
    im4 = axes[1, 1].pcolormesh(t, f, reduction_ref, shading="gouraud", vmin=-40, vmax=40, cmap="coolwarm")
    axes[1, 1].set_title("Noise Reduction (Reference)")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Frequency (Hz)")
    plt.colorbar(im4, ax=axes[1, 1], label="dB reduction")

    reduction_test = spec_orig - spec_test
    im5 = axes[1, 2].pcolormesh(t, f, reduction_test, shading="gouraud", vmin=-40, vmax=40, cmap="coolwarm")
    axes[1, 2].set_title("Noise Reduction (Test)")
    axes[1, 2].set_xlabel("Time (s)")
    axes[1, 2].set_ylabel("Frequency (Hz)")
    plt.colorbar(im5, ax=axes[1, 2], label="dB reduction")

    fig.suptitle(f"Spectrogram Comparison - {title_suffix}", y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_rms_comparison(original: np.ndarray, reference: np.ndarray, test: np.ndarray, sr: int, output_path: Path, title_suffix: str) -> None:
    orig, ref, x = trim3(original, reference, test)
    rms_orig = compute_rms_energy(orig)
    rms_ref = compute_rms_energy(ref)
    rms_test = compute_rms_energy(x)
    rmin = min(len(rms_orig), len(rms_ref), len(rms_test))
    hop = 2400
    t = np.arange(rmin) * (hop / float(sr))

    fig, axes = plt.subplots(2, 1, figsize=(15, 6))
    axes[0].plot(t, rms_orig[:rmin], "b-", label="Original", alpha=0.8)
    axes[0].plot(t, rms_ref[:rmin], "r-", label="Reference", alpha=0.8)
    axes[0].plot(t, rms_test[:rmin], "g-", label="Test", alpha=0.8)
    axes[0].set_title(f"RMS Energy Over Time - {title_suffix}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("RMS")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    diff = rms_ref[:rmin] - rms_test[:rmin]
    axes[1].plot(t, diff, color="purple", label="Reference - Test")
    axes[1].axhline(y=0.0, color="k", linestyle="--", alpha=0.5)
    axes[1].set_title(f"RMS Difference - {title_suffix}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("RMS diff")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_correlation_scatter(reference: np.ndarray, test: np.ndarray, output_path: Path, title_suffix: str) -> None:
    ref, x = trim3(reference, test, test)[:2]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    min_len = min(len(ref), len(x))
    step = max(1, min_len // 10000)
    ref_plot = ref[:min_len:step]
    x_plot = x[:min_len:step]

    axes[0].scatter(ref_plot, x_plot, alpha=0.1, s=1)
    axes[0].plot([-1, 1], [-1, 1], "r--", label="y=x")
    axes[0].set_xlabel("Reference")
    axes[0].set_ylabel("Test")
    axes[0].set_title(f"Sample Correlation - {title_suffix}")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_aspect("equal")

    diff = ref[:min_len] - x[:min_len]
    axes[1].hist(diff, bins=100, density=True, alpha=0.7, color="purple")
    axes[1].axvline(x=0.0, color="r", linestyle="--", label="Zero")
    axes[1].axvline(x=float(np.mean(diff)), color="g", linestyle="-", label=f"Mean: {float(np.mean(diff)):.6f}")
    axes[1].set_xlabel("Difference (Reference - Test)")
    axes[1].set_ylabel("Density")
    axes[1].set_title(f"Difference Distribution - {title_suffix}")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_noise_reduction_overlay(original: np.ndarray, reference: np.ndarray, test: np.ndarray, sr: int, output_path: Path, title_suffix: str) -> None:
    orig, ref, x = trim3(original, reference, test)
    f, t_orig, spec_orig = stft_with_axes(orig, sr)
    _, _, spec_ref = stft_with_axes(ref, sr)
    _, _, spec_test = stft_with_axes(x, sr)
    tmin = min(spec_orig.shape[1], spec_ref.shape[1], spec_test.shape[1])
    t = t_orig[:tmin]
    spec_orig = spec_orig[:, :tmin]
    spec_ref = spec_ref[:, :tmin]
    spec_test = spec_test[:, :tmin]
    reduction_ref = spec_orig - spec_ref
    reduction_test = spec_orig - spec_test

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    vmin, vmax = -40, 40
    axes[0].pcolormesh(t, f, reduction_ref, shading="gouraud", vmin=vmin, vmax=vmax, cmap="Reds", alpha=0.45)
    axes[0].pcolormesh(t, f, reduction_test, shading="gouraud", vmin=vmin, vmax=vmax, cmap="Blues", alpha=0.45)
    axes[0].set_title(f"Noise Reduction Overlay - {title_suffix}")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Frequency (Hz)")
    axes[0].legend(
        handles=[
            Patch(facecolor="red", alpha=0.45, label="Reference"),
            Patch(facecolor="blue", alpha=0.45, label="Test"),
        ],
        loc="upper right",
    )

    reduction_diff = reduction_ref - reduction_test
    im = axes[1].pcolormesh(t, f, reduction_diff, shading="gouraud", vmin=-20, vmax=20, cmap="RdBu_r")
    axes[1].set_title(f"Noise Reduction Diff (Reference - Test) - {title_suffix}")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Frequency (Hz)")
    plt.colorbar(im, ax=axes[1], label="dB reduction diff")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_noise_reduction_absdiff(original: np.ndarray, reference: np.ndarray, test: np.ndarray, sr: int, output_path: Path, title_suffix: str) -> None:
    orig, ref, x = trim3(original, reference, test)
    f, t_orig, spec_orig = stft_with_axes(orig, sr)
    _, _, spec_ref = stft_with_axes(ref, sr)
    _, _, spec_test = stft_with_axes(x, sr)
    tmin = min(spec_orig.shape[1], spec_ref.shape[1], spec_test.shape[1])
    t = t_orig[:tmin]
    spec_orig = spec_orig[:, :tmin]
    spec_ref = spec_ref[:, :tmin]
    spec_test = spec_test[:, :tmin]
    reduction_diff_abs = np.abs((spec_orig - spec_ref) - (spec_orig - spec_test))
    vmax = float(np.percentile(reduction_diff_abs, 99.5))
    if vmax <= 0:
        vmax = 1.0

    plt.figure(figsize=(10, 5))
    im = plt.pcolormesh(t, f, reduction_diff_abs, shading="gouraud", cmap="gray", vmin=0.0, vmax=vmax)
    plt.title(f"Noise Reduction Absolute Diff - {title_suffix}")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(im, label="|Reference - Test| dB")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_timing(results: Dict[str, dict], out_png: Path, key: str, title: str) -> None:
    rows = [(k, v) for k, v in results.items() if v.get("status") == "ok" and v.get(key) is not None]
    if not rows:
        return
    labels = [k for k, _ in rows]
    values = [float(v[key]) for _, v in rows]
    colors = ["#1f77b4" if v["mode"] == "offline" else "#ff7f0e" for _, v in rows]

    plt.figure(figsize=(12, 5))
    bars = plt.bar(labels, values, color=colors)
    plt.ylabel("seconds")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_speedup_vs_new_swift(results: Dict[str, dict], out_png: Path) -> None:
    mode_base: Dict[str, float] = {}
    for name, item in results.items():
        if item.get("status") != "ok":
            continue
        if name == "new_swift_offline":
            mode_base["offline"] = float(item["best_sec"])
        if name == "new_swift_stream_10ms":
            mode_base["stream"] = float(item["best_sec"])

    rows: List[Tuple[str, float]] = []
    for name, item in results.items():
        if item.get("status") != "ok" or item.get("best_sec") is None:
            continue
        base = mode_base.get(item["mode"])
        if base is None:
            continue
        rows.append((name, float(item["best_sec"]) / base))
    if not rows:
        return

    labels = [x[0] for x in rows]
    values = [x[1] for x in rows]
    plt.figure(figsize=(12, 5))
    bars = plt.bar(labels, values, color="#2ca02c")
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.ylabel("best_sec / new_swift_best_sec")
    plt.title("Relative Runtime Vs New Swift (Lower Is Faster)")
    plt.xticks(rotation=25, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_quality(results: Dict[str, dict], out_png: Path) -> None:
    rows = [(k, v) for k, v in results.items() if v.get("status") == "ok" and "correlation" in v and "pearson" in v["correlation"]]
    if not rows:
        return
    labels = [k for k, _ in rows]
    pearson = [float(v["correlation"]["pearson"]) for _, v in rows]
    snr = [float(v["correlation"]["snr_db"]) for _, v in rows]
    x = np.arange(len(labels))
    width = 0.4

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.bar(x, pearson, width, color="#9467bd")
    ax1.set_ylabel("Pearson")
    ax1.set_title("Correlation/Quality Metrics")
    ax1.set_ylim(min(0.8, min(pearson) - 0.01), 1.001)

    ax2.bar(x, snr, width, color="#17becf")
    ax2.set_ylabel("SNR dB")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=25, ha="right")
    ax2.set_xlabel("implementation")

    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_waveform_diff(ref: np.ndarray, x: np.ndarray, sr: int, out_png: Path) -> None:
    n = min(len(ref), len(x), sr)
    t = np.arange(n) / float(sr)
    diff = x[:n] - ref[:n]
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    axes[0].plot(t, ref[:n], label="reference (pytorch_cli)", alpha=0.8)
    axes[0].plot(t, x[:n], label="new_swift", alpha=0.7)
    axes[0].legend()
    axes[0].set_ylabel("amplitude")
    axes[0].set_title("Waveform Overlay (First 1s)")
    axes[1].plot(t, diff, color="red", alpha=0.8)
    axes[1].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("seconds")
    axes[1].set_ylabel("difference")
    axes[1].set_title("new_swift - reference")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def plot_spectrogram_diff(ref: np.ndarray, x: np.ndarray, out_png: Path) -> None:
    n = min(len(ref), len(x))
    ref_db = stft_mag_db(ref[:n])
    x_db = stft_mag_db(x[:n])
    t = min(ref_db.shape[1], x_db.shape[1])
    diff = x_db[:, :t] - ref_db[:, :t]
    plt.figure(figsize=(12, 5))
    plt.imshow(diff, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-20, vmax=20)
    plt.colorbar(label="dB diff (new_swift - reference)")
    plt.title("Spectrogram Difference")
    plt.xlabel("frame")
    plt.ylabel("freq bin")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()


def build_specs(args: argparse.Namespace, compare_dir: Path) -> List[RunSpec]:
    input_wav = Path(args.input).resolve()
    mlx_py_repo = Path(args.mlx_python_repo).resolve()
    mlx_swift_repo = Path(args.mlx_swift_repo).resolve()
    new_swift_repo = Path(args.new_swift_repo).resolve()

    copy_metallib_bundle(mlx_swift_repo)
    copy_metallib_bundle(new_swift_repo)

    pytorch_tmp = Path(tempfile.mkdtemp(prefix="dfn_pytorch_"))
    rust_tmp = Path(tempfile.mkdtemp(prefix="dfn_rust_"))
    ensure_dir(pytorch_tmp)
    ensure_dir(rust_tmp)

    specs: List[RunSpec] = []
    specs.append(
        RunSpec(
            name="pytorch_cli",
            mode="offline",
            cmd=[
                args.py_venv,
                "-m",
                "df.enhance",
                "-m",
                args.pytorch_model_dir,
                "--output-dir",
                str(pytorch_tmp),
                "--no-suffix",
                str(input_wav),
            ],
            raw_output=pytorch_tmp / input_wav.name,
            final_output=compare_dir / "pytorch_cli.wav",
        )
    )
    specs.append(
        RunSpec(
            name="rust_deep_filter",
            mode="offline",
            cmd=[
                args.rust_bin,
                "--output-dir",
                str(rust_tmp),
                str(input_wav),
            ],
            raw_output=rust_tmp / input_wav.name,
            final_output=compare_dir / "rust_deep_filter.wav",
        )
    )
    specs.append(
        RunSpec(
            name="mlx_python_offline",
            mode="offline",
            cwd=mlx_py_repo,
            cmd=[
                args.py_venv,
                "-m",
                "mlx_audio.sts.deepfilternet",
                str(input_wav),
                "--model",
                args.mlx_model_dir,
                "-o",
                str(compare_dir / "mlx_python_offline.wav"),
            ],
            raw_output=compare_dir / "mlx_python_offline.wav",
            final_output=compare_dir / "mlx_python_offline.wav",
        )
    )
    specs.append(
        RunSpec(
            name="mlx_audio_swift_offline",
            mode="offline",
            cwd=mlx_swift_repo,
            cmd=[
                "./.build/release/mlx-audio-swift-sts",
                "--model",
                args.mlx_model_dir,
                "--audio",
                str(input_wav),
                "--mode",
                "short",
                "--output-target",
                str(compare_dir / "mlx_audio_swift_offline.wav"),
            ],
            raw_output=compare_dir / "mlx_audio_swift_offline.wav",
            final_output=compare_dir / "mlx_audio_swift_offline.wav",
        )
    )
    specs.append(
        RunSpec(
            name="new_swift_offline",
            mode="offline",
            cwd=new_swift_repo,
            cmd=[
                "./.build/release/deepfilternet-mlx",
                str(input_wav),
                "--model",
                args.mlx_model_dir,
                "--output",
                str(compare_dir / "new_swift_offline.wav"),
            ],
            raw_output=compare_dir / "new_swift_offline.wav",
            final_output=compare_dir / "new_swift_offline.wav",
        )
    )
    if not args.skip_fp16:
        specs.append(
            RunSpec(
                name="new_swift_offline_fp16",
                mode="offline",
                cwd=new_swift_repo,
                cmd=[
                    "./.build/release/deepfilternet-mlx",
                    str(input_wav),
                    "--model",
                    args.mlx_model_dir,
                    "--precision",
                    "fp16",
                    "--output",
                    str(compare_dir / "new_swift_offline_fp16.wav"),
                ],
                raw_output=compare_dir / "new_swift_offline_fp16.wav",
                final_output=compare_dir / "new_swift_offline_fp16.wav",
            )
        )
    specs.append(
        RunSpec(
            name="mlx_python_stream_10ms",
            mode="stream",
            cwd=mlx_py_repo,
            cmd=[
                args.py_venv,
                "-m",
                "mlx_audio.sts.deepfilternet",
                str(input_wav),
                "--model",
                args.mlx_model_dir,
                "-o",
                str(compare_dir / "mlx_python_stream_10ms.wav"),
                "--stream",
                "--chunk-ms",
                "10",
            ],
            raw_output=compare_dir / "mlx_python_stream_10ms.wav",
            final_output=compare_dir / "mlx_python_stream_10ms.wav",
        )
    )
    specs.append(
        RunSpec(
            name="mlx_audio_swift_stream_10ms",
            mode="stream",
            cwd=mlx_swift_repo,
            cmd=[
                "./.build/release/mlx-audio-swift-sts",
                "--model",
                args.mlx_model_dir,
                "--audio",
                str(input_wav),
                "--mode",
                "stream",
                "--chunk-seconds",
                "0.01",
                "--output-target",
                str(compare_dir / "mlx_audio_swift_stream_10ms.wav"),
            ],
            raw_output=compare_dir / "mlx_audio_swift_stream_10ms.wav",
            final_output=compare_dir / "mlx_audio_swift_stream_10ms.wav",
        )
    )
    specs.append(
        RunSpec(
            name="new_swift_stream_10ms",
            mode="stream",
            cwd=new_swift_repo,
            cmd=[
                "./.build/release/deepfilternet-mlx",
                str(input_wav),
                "--model",
                args.mlx_model_dir,
                "--stream",
                "--chunk-ms",
                "10",
                "--materialize-every-hops",
                str(args.new_swift_materialize_hops),
                "--output",
                str(compare_dir / "new_swift_stream_10ms.wav"),
            ],
            raw_output=compare_dir / "new_swift_stream_10ms.wav",
            final_output=compare_dir / "new_swift_stream_10ms.wav",
        )
    )
    if not args.skip_fp16:
        specs.append(
            RunSpec(
                name="new_swift_stream_10ms_fp16",
                mode="stream",
                cwd=new_swift_repo,
                cmd=[
                    "./.build/release/deepfilternet-mlx",
                    str(input_wav),
                    "--model",
                    args.mlx_model_dir,
                    "--precision",
                    "fp16",
                    "--stream",
                    "--chunk-ms",
                    "10",
                    "--materialize-every-hops",
                    str(args.new_swift_materialize_hops),
                    "--output",
                    str(compare_dir / "new_swift_stream_10ms_fp16.wav"),
                ],
                raw_output=compare_dir / "new_swift_stream_10ms_fp16.wav",
                final_output=compare_dir / "new_swift_stream_10ms_fp16.wav",
            )
        )
    return specs


def write_compare_md(compare_path: Path, reference_name: str, results: Dict[str, dict], quality_gate: Optional[Dict[str, dict]] = None) -> None:
    lines = [
        "# DeepFilterNet Cross-Implementation Comparison",
        "",
        f"Reference: `{reference_name}`",
        "",
        "| Impl | Mode | Status | Pearson | Cosine | SNR(dB) | MAE | MSE | Lag |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in results.items():
        corr = item.get("correlation", {})
        lines.append(
            "| {name} | {mode} | {status} | {pearson} | {cosine} | {snr} | {mae} | {mse} | {lag} |".format(
                name=name,
                mode=item.get("mode", "-"),
                status=item.get("status", "-"),
                pearson=(f"{corr.get('pearson', float('nan')):.6f}" if "pearson" in corr else "-"),
                cosine=(f"{corr.get('cosine', float('nan')):.6f}" if "cosine" in corr else "-"),
                snr=(f"{corr.get('snr_db', float('nan')):.3f}" if "snr_db" in corr else "-"),
                mae=(f"{corr.get('mae', float('nan')):.6e}" if "mae" in corr else "-"),
                mse=(f"{corr.get('mse', float('nan')):.6e}" if "mse" in corr else "-"),
                lag=str(corr.get("lag_samples", "-")),
            )
        )
    if quality_gate and quality_gate.get("targets"):
        lines.extend(
            [
                "",
                "## FP16 Quality Gate",
                "",
                "| Target | Status | Pearson | SNR(dB) | MAE | Spec MAE(dB) | RMS Trace MAE |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for target, gate in quality_gate["targets"].items():
            metrics = gate.get("metrics", {})
            lines.append(
                "| {target} | {status} | {pearson} | {snr} | {mae} | {spec_mae} | {rms_mae} |".format(
                    target=target,
                    status=gate.get("status", "missing"),
                    pearson=(f"{metrics.get('pearson', float('nan')):.6f}" if "pearson" in metrics else "-"),
                    snr=(f"{metrics.get('snr_db', float('nan')):.3f}" if "snr_db" in metrics else "-"),
                    mae=(f"{metrics.get('mae', float('nan')):.6e}" if "mae" in metrics else "-"),
                    spec_mae=(f"{metrics.get('spectral_mae_db', float('nan')):.6f}" if "spectral_mae_db" in metrics else "-"),
                    rms_mae=(f"{metrics.get('rms_trace_mae', float('nan')):.6e}" if "rms_trace_mae" in metrics else "-"),
                )
            )

    compare_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_benchmark_md(benchmark_path: Path, results: Dict[str, dict]) -> None:
    lines = [
        "# DeepFilterNet Benchmark Timing",
        "",
        "| Impl | Mode | Status | Best(s) | Avg(s) |",
        "|---|---|---|---:|---:|",
    ]
    for name, item in results.items():
        lines.append(
            "| {name} | {mode} | {status} | {best} | {avg} |".format(
                name=name,
                mode=item.get("mode", "-"),
                status=item.get("status", "-"),
                best=(f"{item['best_sec']:.3f}" if item.get("best_sec") is not None else "-"),
                avg=(f"{item['avg_sec']:.3f}" if item.get("avg_sec") is not None else "-"),
            )
        )
    benchmark_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_detailed_compare_md(path: Path, detailed: Dict[str, Dict[str, float]], quality_gate: Optional[Dict[str, dict]] = None) -> None:
    lines = [
        "# Detailed Compare Metrics",
        "",
        "| Target | Correlation | SER(dB) | MAE | RMSE | Spec MAE(dB) | Spec MAE >-60dB | Spec p95 | NR Diff MAE(dB) | RMS Trace MAE |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for target, m in detailed.items():
        lines.append(
            "| {target} | {corr:.6f} | {ser:.3f} | {mae:.6e} | {rmse:.6e} | {smae:.6f} | {smae60:.6f} | {sp95:.6f} | {nrmae:.6f} | {rmsmae:.6e} |".format(
                target=target,
                corr=float(m.get("correlation", float("nan"))),
                ser=float(m.get("ser_db", float("nan"))),
                mae=float(m.get("mae", float("nan"))),
                rmse=float(m.get("rmse", float("nan"))),
                smae=float(m.get("spectral_mae_db", float("nan"))),
                smae60=float(m.get("spectral_mae_db_active_gt_-60db", float("nan"))),
                sp95=float(m.get("spectral_p95_abs_db", float("nan"))),
                nrmae=float(m.get("noise_reduction_diff_mae_db", float("nan"))),
                rmsmae=float(m.get("rms_trace_mae", float("nan"))),
            )
        )
    if quality_gate and quality_gate.get("targets"):
        lines.extend(["", "## FP16 Quality Gate Details", ""])
        for target, gate in quality_gate["targets"].items():
            lines.append(f"- `{target}`: `{gate.get('status', 'missing')}`")
            for check_name, check in gate.get("checks", {}).items():
                lines.append(
                    "- {name}: pass={passed} value={value} threshold={threshold}".format(
                        name=check_name,
                        passed=check.get("pass", False),
                        value=check.get("value"),
                        threshold=check.get("threshold"),
                    )
                )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate_fp16_quality_gate(results: Dict[str, dict], detailed: Dict[str, Dict[str, float]]) -> Dict[str, dict]:
    targets = ["new_swift_offline_fp16", "new_swift_stream_10ms_fp16"]
    fp32_baseline = {
        "new_swift_offline_fp16": "new_swift_offline",
        "new_swift_stream_10ms_fp16": "new_swift_stream_10ms",
    }
    thresholds = {
        "pearson_min_report": 0.999,
        "snr_db_min_report": 35.0,
        "mae_max_report": 5e-4,
        "spectral_mae_db_max_report": 1.0,
        "rms_trace_mae_max_report": 2e-4,
        "pearson_drop_max_vs_fp32": 5e-4,
        "snr_drop_db_max_vs_fp32": 2.5,
        "mae_increase_max_vs_fp32": 2e-4,
        "spectral_mae_db_increase_max_vs_fp32": 0.3,
    }

    gate_targets: Dict[str, dict] = {}
    for target in targets:
        target_item = results.get(target)
        if not target_item or target_item.get("status") != "ok":
            gate_targets[target] = {"status": "missing", "checks": {}, "metrics": {}}
            continue

        corr = target_item.get("correlation", {})
        det = detailed.get(target, {})
        baseline_name = fp32_baseline[target]
        base_item = results.get(baseline_name, {})
        base_corr = base_item.get("correlation", {}) if base_item.get("status") == "ok" else {}
        base_det = detailed.get(baseline_name, {})

        metrics = {
            "pearson": float(corr.get("pearson", float("nan"))),
            "snr_db": float(corr.get("snr_db", float("nan"))),
            "mae": float(corr.get("mae", float("nan"))),
            "spectral_mae_db": float(det.get("spectral_mae_db", float("nan"))),
            "rms_trace_mae": float(det.get("rms_trace_mae", float("nan"))),
        }

        report_checks = {
            "pearson_min_report": {
                "pass": metrics["pearson"] >= thresholds["pearson_min_report"],
                "value": metrics["pearson"],
                "threshold": thresholds["pearson_min_report"],
            },
            "snr_db_min_report": {
                "pass": metrics["snr_db"] >= thresholds["snr_db_min_report"],
                "value": metrics["snr_db"],
                "threshold": thresholds["snr_db_min_report"],
            },
            "mae_max_report": {
                "pass": metrics["mae"] <= thresholds["mae_max_report"],
                "value": metrics["mae"],
                "threshold": thresholds["mae_max_report"],
            },
            "spectral_mae_db_max_report": {
                "pass": metrics["spectral_mae_db"] <= thresholds["spectral_mae_db_max_report"],
                "value": metrics["spectral_mae_db"],
                "threshold": thresholds["spectral_mae_db_max_report"],
            },
            "rms_trace_mae_max_report": {
                "pass": metrics["rms_trace_mae"] <= thresholds["rms_trace_mae_max_report"],
                "value": metrics["rms_trace_mae"],
                "threshold": thresholds["rms_trace_mae_max_report"],
            },
        }
        gate_checks = dict(report_checks)
        used_checks = report_checks

        if base_corr:
            pearson_drop = float(base_corr.get("pearson", 0.0)) - metrics["pearson"]
            snr_drop = float(base_corr.get("snr_db", 0.0)) - metrics["snr_db"]
            mae_inc = metrics["mae"] - float(base_corr.get("mae", 0.0))
            spec_mae_inc = metrics["spectral_mae_db"] - float(base_det.get("spectral_mae_db", 0.0))
            delta_checks = {
                "pearson_drop_max_vs_fp32": {
                    "pass": pearson_drop <= thresholds["pearson_drop_max_vs_fp32"],
                    "value": pearson_drop,
                    "threshold": thresholds["pearson_drop_max_vs_fp32"],
                },
                "snr_drop_db_max_vs_fp32": {
                    "pass": snr_drop <= thresholds["snr_drop_db_max_vs_fp32"],
                    "value": snr_drop,
                    "threshold": thresholds["snr_drop_db_max_vs_fp32"],
                },
                "mae_increase_max_vs_fp32": {
                    "pass": mae_inc <= thresholds["mae_increase_max_vs_fp32"],
                    "value": mae_inc,
                    "threshold": thresholds["mae_increase_max_vs_fp32"],
                },
                "spectral_mae_db_increase_max_vs_fp32": {
                    "pass": spec_mae_inc <= thresholds["spectral_mae_db_increase_max_vs_fp32"],
                    "value": spec_mae_inc,
                    "threshold": thresholds["spectral_mae_db_increase_max_vs_fp32"],
                },
            }
            gate_checks.update(delta_checks)
            used_checks = delta_checks

        status = "pass" if all(bool(v.get("pass", False)) for v in used_checks.values()) else "fail"
        gate_targets[target] = {
            "status": status,
            "checks": gate_checks,
            "gate_basis": "delta_vs_fp32" if base_corr else "absolute_report_only",
            "metrics": metrics,
            "baseline": baseline_name,
        }

    return {"thresholds": thresholds, "targets": gate_targets}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run latest DeepFilterNet compare+benchmark and overwrite fixed output files")
    parser.add_argument("--input", required=True)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--outputs-root", default="/Users/kylehowells/Developer/Github/DeepFilterNet-mlx/outputs")
    parser.add_argument(
        "--py-venv",
        default="/Users/kylehowells/Developer/Example-Projects/mlx-audio-master-codex/.venv/bin/python",
    )
    parser.add_argument(
        "--pytorch-model-dir",
        default="/Users/kylehowells/Developer/Example-Projects/DeepFilterNet/DeepFilterNet/models/extracted/DeepFilterNet3",
    )
    parser.add_argument(
        "--rust-bin",
        default="/Users/kylehowells/Developer/Example-Projects/DeepFilterNet/DeepFilterNet/target/release/deep-filter",
    )
    parser.add_argument(
        "--mlx-model-dir",
        default="/Users/kylehowells/Developer/Example-Projects/mlx-audio-master-codex/models/DeepFilterNet3",
    )
    parser.add_argument(
        "--mlx-python-repo",
        default="/Users/kylehowells/Developer/Example-Projects/mlx-audio-master-codex",
    )
    parser.add_argument(
        "--mlx-swift-repo",
        default="/Users/kylehowells/Developer/Example-Projects/mlx-audio-swift-master",
    )
    parser.add_argument(
        "--new-swift-repo",
        default="/Users/kylehowells/Developer/Github/DeepFilterNet-mlx",
    )
    parser.add_argument("--new-swift-materialize-hops", type=int, default=96)
    parser.add_argument("--skip-fp16", action="store_true", help="Skip FP16 new_swift benchmark variants")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).resolve()
    compare_dir = outputs_root / "compare"
    benchmark_dir = outputs_root / "benchmark"
    ensure_dir(compare_dir)
    ensure_dir(benchmark_dir)

    specs = build_specs(args, compare_dir)
    results: Dict[str, dict] = {}
    for spec in specs:
        times, error = run_cmd(spec, repeats=args.repeats)
        results[spec.name] = {
            "name": spec.name,
            "mode": spec.mode,
            "output": str(spec.final_output),
            "times_sec": times,
            "best_sec": min(times) if times else None,
            "avg_sec": float(np.mean(times)) if times else None,
            "status": "ok" if error is None else "failed",
            "error": error,
        }

    reference_name = "pytorch_cli" if results.get("pytorch_cli", {}).get("status") == "ok" else "rust_deep_filter"
    ref_audio, ref_sr = load_mono(Path(results[reference_name]["output"]))
    for name, item in results.items():
        if item["status"] != "ok":
            continue
        y, sr = load_mono(Path(item["output"]))
        if sr != ref_sr:
            y = resample_linear(y, sr, ref_sr)
            sr = ref_sr
        ref_aligned, y_aligned, lag = align_signals(ref_audio, y)
        corr = metrics_vs_ref(ref_aligned, y_aligned)
        corr["lag_samples"] = lag
        corr["aligned_len"] = int(len(ref_aligned))
        item["correlation"] = corr

    compare_json = compare_dir / "comparison_results.json"
    compare_md = compare_dir / "comparison_results.md"
    detailed_json = compare_dir / "detailed_metrics.json"
    detailed_md = compare_dir / "detailed_metrics.md"
    benchmark_json = benchmark_dir / "benchmark_results.json"
    benchmark_md = benchmark_dir / "benchmark_results.md"

    plot_timing(results, benchmark_dir / "timing_best_sec.png", "best_sec", "Benchmark (Best Runtime)")
    plot_timing(results, benchmark_dir / "timing_avg_sec.png", "avg_sec", "Benchmark (Average Runtime)")
    plot_speedup_vs_new_swift(results, benchmark_dir / "speedup_vs_new_swift.png")
    plot_quality(results, compare_dir / "quality_metrics.png")

    detailed_metrics: Dict[str, Dict[str, float]] = {}
    input_audio, input_sr = load_mono(Path(args.input))
    if input_sr != ref_sr:
        input_audio = resample_linear(input_audio, input_sr, ref_sr)
        input_sr = ref_sr

    generated_pngs: List[Path] = []

    def generate_detailed_artifacts(target_name: str, make_latest_alias: bool = False) -> None:
        if results.get(target_name, {}).get("status") != "ok":
            return
        new_audio, new_sr = load_mono(Path(results[target_name]["output"]))
        if new_sr != ref_sr:
            new_audio = resample_linear(new_audio, new_sr, ref_sr)
            new_sr = ref_sr

        ref_aligned, new_aligned, _ = align_signals(ref_audio, new_audio)
        waveform_vs = compare_dir / f"{target_name}_vs_reference_waveform.png"
        spectrogram_vs = compare_dir / f"{target_name}_vs_reference_spectrogram_diff.png"
        plot_waveform_diff(ref_aligned, new_aligned, ref_sr, waveform_vs)
        plot_spectrogram_diff(ref_aligned, new_aligned, spectrogram_vs)
        generated_pngs.extend([waveform_vs, spectrogram_vs])

        if make_latest_alias:
            alias_wave = compare_dir / "new_swift_vs_reference_waveform.png"
            alias_spec = compare_dir / "new_swift_vs_reference_spectrogram_diff.png"
            shutil.copy2(waveform_vs, alias_wave)
            shutil.copy2(spectrogram_vs, alias_spec)
            generated_pngs.extend([alias_wave, alias_spec])

        orig_s, ref_s, new_s = trim3(input_audio, ref_aligned, new_aligned)
        detailed_metrics[target_name] = compute_detailed_metrics(orig_s, ref_s, new_s, ref_sr)

        waveform_cmp = compare_dir / f"waveform_comparison_{target_name}.png"
        spectrogram_cmp = compare_dir / f"spectrogram_comparison_{target_name}.png"
        rms_cmp = compare_dir / f"rms_comparison_{target_name}.png"
        corr_scatter = compare_dir / f"correlation_scatter_{target_name}.png"
        nr_overlay = compare_dir / f"noise_reduction_overlay_{target_name}.png"
        nr_absdiff = compare_dir / f"noise_reduction_absdiff_{target_name}.png"

        plot_waveform_comparison(orig_s, ref_s, new_s, ref_sr, waveform_cmp, target_name)
        plot_spectrogram_comparison(orig_s, ref_s, new_s, ref_sr, spectrogram_cmp, target_name)
        plot_rms_comparison(orig_s, ref_s, new_s, ref_sr, rms_cmp, target_name)
        plot_correlation_scatter(ref_s, new_s, corr_scatter, target_name)
        plot_noise_reduction_overlay(orig_s, ref_s, new_s, ref_sr, nr_overlay, target_name)
        plot_noise_reduction_absdiff(orig_s, ref_s, new_s, ref_sr, nr_absdiff, target_name)
        generated_pngs.extend([waveform_cmp, spectrogram_cmp, rms_cmp, corr_scatter, nr_overlay, nr_absdiff])

    detail_targets = ["new_swift_stream_10ms", "new_swift_offline"]
    if not args.skip_fp16:
        detail_targets.extend(["new_swift_stream_10ms_fp16", "new_swift_offline_fp16"])

    for target_name in detail_targets:
        generate_detailed_artifacts(target_name, make_latest_alias=(target_name == "new_swift_stream_10ms"))

    quality_gate = evaluate_fp16_quality_gate(results, detailed_metrics) if not args.skip_fp16 else {"targets": {}, "thresholds": {}}

    compare_json.write_text(
        json.dumps({"reference": reference_name, "results": results, "quality_gate": quality_gate}, indent=2),
        encoding="utf-8",
    )
    benchmark_json.write_text(
        json.dumps(
            {
                "repeats": args.repeats,
                "timings": {
                    k: {
                        "mode": v["mode"],
                        "status": v["status"],
                        "best_sec": v["best_sec"],
                        "avg_sec": v["avg_sec"],
                        "times_sec": v["times_sec"],
                    }
                    for k, v in results.items()
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_compare_md(compare_md, reference_name, results, quality_gate=quality_gate)
    write_benchmark_md(benchmark_md, results)
    detailed_json.write_text(
        json.dumps({"metrics": detailed_metrics, "quality_gate": quality_gate}, indent=2),
        encoding="utf-8",
    )
    write_detailed_compare_md(detailed_md, detailed_metrics, quality_gate=quality_gate)

    print(f"Wrote {compare_json}")
    print(f"Wrote {compare_md}")
    print(f"Wrote {detailed_json}")
    print(f"Wrote {detailed_md}")
    print(f"Wrote {benchmark_json}")
    print(f"Wrote {benchmark_md}")
    print(f"Wrote {compare_dir / 'quality_metrics.png'}")
    print(f"Wrote {compare_dir / 'new_swift_vs_reference_waveform.png'}")
    print(f"Wrote {compare_dir / 'new_swift_vs_reference_spectrogram_diff.png'}")
    print(f"Wrote {compare_dir / 'new_swift_stream_10ms_vs_reference_waveform.png'}")
    print(f"Wrote {compare_dir / 'new_swift_stream_10ms_vs_reference_spectrogram_diff.png'}")
    print(f"Wrote {compare_dir / 'new_swift_offline_vs_reference_waveform.png'}")
    print(f"Wrote {compare_dir / 'new_swift_offline_vs_reference_spectrogram_diff.png'}")
    for png in sorted(set(generated_pngs)):
        print(f"Wrote {png}")
    print(f"Wrote {benchmark_dir / 'timing_best_sec.png'}")
    print(f"Wrote {benchmark_dir / 'timing_avg_sec.png'}")
    print(f"Wrote {benchmark_dir / 'speedup_vs_new_swift.png'}")


if __name__ == "__main__":
    main()
