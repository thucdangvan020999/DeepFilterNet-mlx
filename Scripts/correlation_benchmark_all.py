#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import soundfile as sf
except Exception as exc:  # pragma: no cover
    raise RuntimeError("soundfile is required for this script") from exc


@dataclass
class RunSpec:
    name: str
    mode: str  # offline | stream
    cmd: List[str]
    output: Path
    cwd: Optional[Path] = None
    env: Optional[Dict[str, str]] = None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def run_cmd(spec: RunSpec, repeats: int) -> Tuple[List[float], Optional[str]]:
    times: List[float] = []
    error: Optional[str] = None

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

        if not spec.output.exists():
            error = (
                f"Output missing for {spec.name}: {spec.output}\n"
                f"command={' '.join(spec.cmd)}\n"
                f"output:\n{proc.stdout[-4000:]}"
            )
            break

        times.append(dt)

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
    ref = ref.astype(np.float64)
    x = x.astype(np.float64)

    ref_mean = ref.mean()
    x_mean = x.mean()

    ref_c = ref - ref_mean
    x_c = x - x_mean

    num = float(np.dot(ref_c, x_c))
    den = float(np.linalg.norm(ref_c) * np.linalg.norm(x_c) + eps)
    pearson = num / den

    cos = float(np.dot(ref, x) / (np.linalg.norm(ref) * np.linalg.norm(x) + eps))

    err = x - ref
    mse = float(np.mean(err * err))
    mae = float(np.mean(np.abs(err)))
    snr = float(10.0 * np.log10((np.mean(ref * ref) + eps) / (mse + eps)))

    return {
        "pearson": pearson,
        "cosine": cos,
        "mse": mse,
        "mae": mae,
        "snr_db": snr,
    }


def copy_metallib_bundle(swift_repo: Path) -> None:
    source = swift_repo / ".xcodebuild/Build/Products/Debug/mlx-swift_Cmlx.bundle"
    dest_parent = swift_repo / ".build/arm64-apple-macosx/debug"
    dest = dest_parent / "mlx-swift_Cmlx.bundle"

    if not source.exists():
        return
    ensure_dir(dest_parent)
    if not dest.exists():
        shutil.copytree(source, dest)


def build_specs(args: argparse.Namespace, out_dir: Path) -> List[RunSpec]:
    input_wav = Path(args.input).resolve()
    ensure_dir(out_dir)

    mlx_py_repo = Path(args.mlx_python_repo).resolve()
    mlx_swift_repo = Path(args.mlx_swift_repo).resolve()
    new_swift_repo = Path(args.new_swift_repo).resolve()

    copy_metallib_bundle(mlx_swift_repo)
    copy_metallib_bundle(new_swift_repo)

    specs: List[RunSpec] = []

    pytorch_out_dir = out_dir / "pytorch"
    ensure_dir(pytorch_out_dir)
    pytorch_out = pytorch_out_dir / input_wav.name
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
                str(pytorch_out_dir),
                "--no-suffix",
                str(input_wav),
            ],
            output=pytorch_out,
        )
    )

    rust_out_dir = out_dir / "rust"
    ensure_dir(rust_out_dir)
    rust_out = rust_out_dir / input_wav.name
    specs.append(
        RunSpec(
            name="rust_deep_filter",
            mode="offline",
            cmd=[
                args.rust_bin,
                "--output-dir",
                str(rust_out_dir),
                str(input_wav),
            ],
            output=rust_out,
        )
    )

    mlx_py_offline = out_dir / "mlx_python_offline.wav"
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
                "-m",
                args.mlx_model_dir,
                "-o",
                str(mlx_py_offline),
            ],
            output=mlx_py_offline,
        )
    )

    mlx_swift_offline = out_dir / "mlx_swift_offline.wav"
    specs.append(
        RunSpec(
            name="mlx_audio_swift_offline",
            mode="offline",
            cwd=mlx_swift_repo,
            cmd=[
                "./.build/debug/mlx-audio-swift-sts",
                "--model",
                args.mlx_model_dir,
                "--audio",
                str(input_wav),
                "--mode",
                "short",
                "--output-target",
                str(mlx_swift_offline),
            ],
            output=mlx_swift_offline,
        )
    )

    new_swift_offline = out_dir / "new_swift_offline.wav"
    specs.append(
        RunSpec(
            name="new_swift_offline",
            mode="offline",
            cwd=new_swift_repo,
            cmd=[
                "./.build/debug/deepfilternet-mlx",
                str(input_wav),
                "--model",
                args.mlx_model_dir,
                "--output",
                str(new_swift_offline),
            ],
            output=new_swift_offline,
        )
    )

    # Streaming-capable implementations
    mlx_py_stream = out_dir / "mlx_python_stream.wav"
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
                "-m",
                args.mlx_model_dir,
                "-o",
                str(mlx_py_stream),
                "--stream",
                "--chunk-ms",
                "10",
            ],
            output=mlx_py_stream,
        )
    )

    mlx_swift_stream = out_dir / "mlx_swift_stream_10ms.wav"
    specs.append(
        RunSpec(
            name="mlx_audio_swift_stream_10ms",
            mode="stream",
            cwd=mlx_swift_repo,
            cmd=[
                "./.build/debug/mlx-audio-swift-sts",
                "--model",
                args.mlx_model_dir,
                "--audio",
                str(input_wav),
                "--mode",
                "stream",
                "--chunk-seconds",
                "0.01",
                "--output-target",
                str(mlx_swift_stream),
            ],
            output=mlx_swift_stream,
        )
    )

    new_swift_stream = out_dir / "new_swift_stream_10ms.wav"
    specs.append(
        RunSpec(
            name="new_swift_stream_10ms",
            mode="stream",
            cwd=new_swift_repo,
            cmd=[
                "./.build/debug/deepfilternet-mlx",
                str(input_wav),
                "--model",
                args.mlx_model_dir,
                "--stream",
                "--chunk-ms",
                "10",
                "--materialize-every-hops",
                str(args.new_swift_materialize_hops),
                "--output",
                str(new_swift_stream),
            ],
            output=new_swift_stream,
        )
    )

    return specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Cross-implementation DeepFilterNet comparison")
    parser.add_argument("--input", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--repeats", type=int, default=2)

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

    args = parser.parse_args()
    out_dir = Path(args.out_dir).resolve()
    ensure_dir(out_dir)

    specs = build_specs(args, out_dir)

    results: Dict[str, dict] = {}
    for spec in specs:
        times, error = run_cmd(spec, repeats=args.repeats)
        result = {
            "name": spec.name,
            "mode": spec.mode,
            "output": str(spec.output),
            "times_sec": times,
            "best_sec": min(times) if times else None,
            "avg_sec": float(np.mean(times)) if times else None,
            "status": "ok" if error is None else "failed",
            "error": error,
        }
        results[spec.name] = result

    # Choose reference for correlation
    reference_name = "pytorch_cli" if results.get("pytorch_cli", {}).get("status") == "ok" else "rust_deep_filter"
    ref_path = Path(results[reference_name]["output"])
    ref_audio, ref_sr = load_mono(ref_path)

    for name, item in results.items():
        if item["status"] != "ok":
            continue
        audio_path = Path(item["output"])
        y, sr = load_mono(audio_path)

        if sr != ref_sr:
            item["correlation"] = {"error": f"sample rate mismatch: {sr} vs {ref_sr}"}
            continue

        ref_aligned, y_aligned, lag = align_signals(ref_audio, y)
        corr = metrics_vs_ref(ref_aligned, y_aligned)
        corr["lag_samples"] = lag
        corr["aligned_len"] = int(len(ref_aligned))
        item["correlation"] = corr

    json_path = out_dir / "comparison_results.json"
    json_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    md_lines = [
        "# DeepFilterNet Cross-Implementation Results",
        "",
        f"Reference output: `{reference_name}`",
        "",
        "| Impl | Mode | Status | Best(s) | Avg(s) | Pearson | Cosine | SNR(dB) | Lag(samples) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]

    for name, item in results.items():
        corr = item.get("correlation", {})
        md_lines.append(
            "| {name} | {mode} | {status} | {best} | {avg} | {pearson} | {cosine} | {snr} | {lag} |".format(
                name=name,
                mode=item["mode"],
                status=item["status"],
                best=(f"{item['best_sec']:.3f}" if item["best_sec"] is not None else "-"),
                avg=(f"{item['avg_sec']:.3f}" if item["avg_sec"] is not None else "-"),
                pearson=(f"{corr.get('pearson', float('nan')):.6f}" if "pearson" in corr else "-"),
                cosine=(f"{corr.get('cosine', float('nan')):.6f}" if "cosine" in corr else "-"),
                snr=(f"{corr.get('snr_db', float('nan')):.3f}" if "snr_db" in corr else "-"),
                lag=(str(corr.get("lag_samples", "-"))),
            )
        )

    md_path = out_dir / "comparison_results.md"
    md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
