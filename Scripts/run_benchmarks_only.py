#!/usr/bin/env python3
"""Run selected DeepFilterNet benchmarks with per-type folders and adaptive repeats."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from compare_and_benchmark_latest import (
    build_specs,
    ensure_dir,
    plot_speedup_vs_new_swift,
    plot_timing,
    run_cmd,
    write_benchmark_md,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run selected benchmark-only timing for DeepFilterNet implementations")
    parser.add_argument("--input", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--single-run-threshold-sec", type=float, default=30.0)
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
    parser.add_argument("--skip-fp16", action="store_true")
    args = parser.parse_args()

    outputs_root = Path(args.outputs_root).resolve()
    compare_dir = outputs_root / "compare"
    benchmark_dir = outputs_root / "benchmark"
    ensure_dir(compare_dir)
    ensure_dir(benchmark_dir)

    selected_order = [
        "pytorch_cli",
        "rust_deep_filter",
        "mlx_python_offline",
        "new_swift_offline",
        "mlx_python_stream_10ms",
        "new_swift_stream_10ms",
    ]
    specs = {s.name: s for s in build_specs(args, compare_dir)}

    results: Dict[str, dict] = {}
    for name in selected_order:
        if name not in specs:
            results[name] = {
                "name": name,
                "mode": "unknown",
                "output": "",
                "times_sec": [],
                "best_sec": None,
                "avg_sec": None,
                "status": "failed",
                "error": f"missing spec: {name}",
                "runs_completed": 0,
                "runs_requested": args.repeats,
            }
            continue

        spec = specs[name]
        per_type_dir = benchmark_dir / name
        ensure_dir(per_type_dir)

        first_times, first_error = run_cmd(spec, repeats=1)
        times: List[float] = list(first_times)
        error = first_error
        runs_requested = args.repeats

        if error is None and times and times[0] <= args.single_run_threshold_sec and args.repeats > 1:
            extra_times, extra_error = run_cmd(spec, repeats=args.repeats - 1)
            times.extend(extra_times)
            if extra_error is not None:
                error = extra_error
        else:
            runs_requested = 1

        results[spec.name] = {
            "name": spec.name,
            "mode": spec.mode,
            "output": str(spec.final_output),
            "times_sec": times,
            "best_sec": min(times) if times else None,
            "avg_sec": (sum(times) / len(times)) if times else None,
            "status": "ok" if error is None else "failed",
            "error": error,
            "runs_completed": len(times),
            "runs_requested": runs_requested,
        }
        (per_type_dir / "stats.json").write_text(json.dumps(results[spec.name], indent=2) + "\n", encoding="utf-8")

    benchmark_json = benchmark_dir / "benchmark_results.json"
    benchmark_md = benchmark_dir / "benchmark_results.md"
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
                        "runs_completed": v["runs_completed"],
                        "runs_requested": v["runs_requested"],
                    }
                    for k, v in results.items()
                },
                "policy": {
                    "default_repeats": args.repeats,
                    "single_run_threshold_sec": args.single_run_threshold_sec,
                    "single_run_if_first_over_threshold": True,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    write_benchmark_md(benchmark_md, results)
    plot_timing(results, benchmark_dir / "timing_best_sec.png", "best_sec", "Benchmark (Best Runtime)")
    plot_timing(results, benchmark_dir / "timing_avg_sec.png", "avg_sec", "Benchmark (Average Runtime)")
    plot_speedup_vs_new_swift(results, benchmark_dir / "speedup_vs_new_swift.png")

    print(f"Wrote {benchmark_json}")
    print(f"Wrote {benchmark_md}")
    print(f"Wrote {benchmark_dir / 'timing_best_sec.png'}")
    print(f"Wrote {benchmark_dir / 'timing_avg_sec.png'}")
    print(f"Wrote {benchmark_dir / 'speedup_vs_new_swift.png'}")


if __name__ == "__main__":
    main()
