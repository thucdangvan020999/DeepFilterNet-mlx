#!/usr/bin/env python3
"""Benchmark DeepFilterNet-mlx CLI against Rust DeepFilterNet binaries.

Usage:
  python Scripts/benchmark_runtime.py \
    --input /path/to/test.wav \
    --model /path/to/DeepFilterNet3 \
    --out-dir /tmp/dfn-bench

Optional Rust commands:
  --rust-offline-cmd "deep-filter {input} -o {output}"
  --rust-stream-cmd "deep-filter-stream {input} -o {output}"
"""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
import wave
from pathlib import Path
from typing import Dict, List


def audio_duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
    if rate <= 0:
        return 0.0
    return frames / float(rate)


def run_cmd(cmd: str) -> float:
    start = time.perf_counter()
    subprocess.run(shlex.split(cmd), check=True)
    return time.perf_counter() - start


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark DeepFilterNet runtimes")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--swift-cmd", type=str, default="swift run deepfilternet-mlx")
    parser.add_argument("--rust-offline-cmd", type=str, default="")
    parser.add_argument("--rust-stream-cmd", type=str, default="")
    parser.add_argument("--stream-chunk-ms", type=float, default=10.0)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    duration = audio_duration_seconds(args.input)
    if duration <= 0:
        raise RuntimeError(f"Could not compute duration for {args.input}")

    scenarios: List[Dict[str, str]] = []

    swift_offline_out = args.out_dir / "swift_offline.wav"
    scenarios.append(
        {
            "name": "swift_offline",
            "cmd": (
                f"{args.swift_cmd} {args.input} --model {args.model} "
                f"--output {swift_offline_out}"
            ),
        }
    )

    swift_stream_out = args.out_dir / "swift_stream.wav"
    scenarios.append(
        {
            "name": "swift_stream",
            "cmd": (
                f"{args.swift_cmd} {args.input} --model {args.model} --stream "
                f"--chunk-ms {args.stream_chunk_ms} --output {swift_stream_out}"
            ),
        }
    )

    if args.rust_offline_cmd:
        scenarios.append(
            {
                "name": "rust_offline",
                "cmd": args.rust_offline_cmd.format(
                    input=args.input,
                    output=args.out_dir / "rust_offline.wav",
                ),
            }
        )

    if args.rust_stream_cmd:
        scenarios.append(
            {
                "name": "rust_stream",
                "cmd": args.rust_stream_cmd.format(
                    input=args.input,
                    output=args.out_dir / "rust_stream.wav",
                ),
            }
        )

    results = []
    for scenario in scenarios:
        runs = []
        for _ in range(max(1, args.repeats)):
            elapsed = run_cmd(scenario["cmd"])
            runs.append(elapsed)

        best = min(runs)
        avg = sum(runs) / len(runs)
        results.append(
            {
                "name": scenario["name"],
                "runs_sec": runs,
                "best_sec": best,
                "avg_sec": avg,
                "best_rtf": best / duration,
                "avg_rtf": avg / duration,
            }
        )

    out_json = args.out_dir / "benchmark_results.json"
    out_json.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print(f"Audio duration: {duration:.3f}s")
    for item in results:
        print(
            f"{item['name']}: best={item['best_sec']:.3f}s "
            f"(RTF={item['best_rtf']:.3f}) avg={item['avg_sec']:.3f}s"
        )
    print(f"Wrote {out_json}")


if __name__ == "__main__":
    main()
