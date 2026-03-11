#!/usr/bin/env python3
"""
Generate benchmark comparison graphs from DeepFilterNet benchmark JSON results.

Usage:
    python Scripts/generate_benchmark_graphs.py benchmark_results/benchmark_2026-03-10_12-00-00.json
    python Scripts/generate_benchmark_graphs.py benchmark_results/  # process all JSON files in dir
"""

import json
import os
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    import numpy as np
except ImportError:
    print("Error: matplotlib and numpy are required.")
    print("Install with: pip install matplotlib numpy")
    sys.exit(1)


# Color palette for engines
ENGINE_COLORS = {
    'mlx_gpu': '#2196F3',          # Blue
    'cpu_accelerate': '#4CAF50',    # Green
    'hybrid_compile_cpu': '#FF9800', # Orange
    'coreml_ane': '#9C27B0',        # Purple
}

ENGINE_ORDER = ['mlx_gpu', 'cpu_accelerate', 'hybrid_compile_cpu', 'coreml_ane']


def load_results(path):
    """Load benchmark results from JSON file."""
    with open(path) as f:
        return json.load(f)


def get_color(engine_id):
    return ENGINE_COLORS.get(engine_id, '#757575')


def plot_streaming_total_time(data, output_dir):
    """Bar chart of total streaming time per engine."""
    results = data.get('streaming_results', [])
    if not results:
        return

    # Sort by engine order
    results.sort(key=lambda r: ENGINE_ORDER.index(r['engine']) if r['engine'] in ENGINE_ORDER else 99)

    names = [r['engine_display'] for r in results]
    times = [r['total_streaming_time_seconds'] for r in results]
    colors = [get_color(r['engine']) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, times, color=colors, edgecolor='white', linewidth=1.5)

    # Add value labels on bars
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.02,
                f'{t:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)

    audio_len = data.get('audio_length_seconds', 0)
    ax.axhline(y=audio_len, color='red', linestyle='--', alpha=0.7, label=f'Real-time ({audio_len:.1f}s)')

    ax.set_ylabel('Total Time (seconds)', fontsize=12)
    ax.set_title(f'Streaming Total Processing Time\n({audio_len:.1f}s audio)', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'streaming_total_time.png'), dpi=150)
    plt.close()


def plot_per_hop_latency(data, output_dir):
    """Bar chart comparing startup vs steady-state per-hop latency."""
    results = data.get('streaming_results', [])
    if not results:
        return

    results.sort(key=lambda r: ENGINE_ORDER.index(r['engine']) if r['engine'] in ENGINE_ORDER else 99)

    names = [r['engine_display'] for r in results]
    startup = [r['startup_per_hop_ms'] for r in results]
    steady = [r['steady_state_per_hop_ms'] for r in results]

    x = np.arange(len(names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width/2, startup, width, label='Startup (first 2 hops)',
                   color=[get_color(r['engine']) for r in results], alpha=0.6, edgecolor='white')
    bars2 = ax.bar(x + width/2, steady, width, label='Steady State (median)',
                   color=[get_color(r['engine']) for r in results], edgecolor='white')

    for bar, v in zip(bars1, startup):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)
    for bar, v in zip(bars2, steady):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9)

    # 10ms real-time line (hop = 480 samples at 48kHz = 10ms)
    ax.axhline(y=10.0, color='red', linestyle='--', alpha=0.7, label='10ms real-time budget')

    ax.set_ylabel('Per-Hop Latency (ms)', fontsize=12)
    ax.set_title('Streaming Per-Hop Latency: Startup vs Steady State', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_hop_latency.png'), dpi=150)
    plt.close()


def plot_hop_timeline(data, output_dir):
    """Line chart of per-hop timing over time for each engine."""
    results = data.get('streaming_results', [])
    if not results:
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    for r in results:
        timings = r.get('per_hop_timings_ms', [])
        if not timings:
            continue
        hops = [t['hop'] for t in timings]
        durations = [t['duration_ms'] for t in timings]

        # Smooth with rolling average for readability
        if len(durations) > 20:
            window = min(20, len(durations) // 5)
            smoothed = np.convolve(durations, np.ones(window)/window, mode='valid')
            hops_smooth = hops[:len(smoothed)]
        else:
            smoothed = durations
            hops_smooth = hops

        ax.plot(hops_smooth, smoothed, label=r['engine_display'],
                color=get_color(r['engine']), linewidth=1.5, alpha=0.8)

    ax.axhline(y=10.0, color='red', linestyle='--', alpha=0.5, label='10ms budget')
    ax.set_xlabel('Hop Index', fontsize=12)
    ax.set_ylabel('Per-Hop Time (ms)', fontsize=12)
    ax.set_title('Per-Hop Processing Time Over Audio Duration', fontsize=14)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'hop_timeline.png'), dpi=150)
    plt.close()


def plot_realtime_factor(data, output_dir):
    """Bar chart of real-time factors."""
    results = data.get('streaming_results', [])
    if not results:
        return

    results.sort(key=lambda r: ENGINE_ORDER.index(r['engine']) if r['engine'] in ENGINE_ORDER else 99)

    names = [r['engine_display'] for r in results]
    rt_factors = [r['realtime_factor'] for r in results]
    colors = [get_color(r['engine']) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, rt_factors, color=colors, edgecolor='white', linewidth=1.5)

    for bar, rt in zip(bars, rt_factors):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.1,
                f'{rt:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='1x real-time (minimum)')

    ax.set_ylabel('Real-Time Factor', fontsize=12)
    ax.set_title('Streaming Real-Time Factor (higher = faster)', fontsize=14)
    ax.legend(fontsize=11)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'realtime_factor.png'), dpi=150)
    plt.close()


def plot_init_time(data, output_dir):
    """Bar chart of initialization time."""
    results = data.get('streaming_results', [])
    if not results:
        return

    results.sort(key=lambda r: ENGINE_ORDER.index(r['engine']) if r['engine'] in ENGINE_ORDER else 99)

    names = [r['engine_display'] for r in results]
    init_times = [r['init_time_seconds'] * 1000 for r in results]
    colors = [get_color(r['engine']) for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, init_times, color=colors, edgecolor='white', linewidth=1.5)

    for bar, t in zip(bars, init_times):
        ax.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                f'{t:.1f}ms', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax.set_ylabel('Initialization Time (ms)', fontsize=12)
    ax.set_title('Engine Initialization Time', fontsize=14)
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'init_time.png'), dpi=150)
    plt.close()


def plot_offline_comparison(data, output_dir):
    """Bar chart of offline processing times."""
    results = data.get('offline_results', [])
    if not results:
        return

    results.sort(key=lambda r: ENGINE_ORDER.index(r['engine']) if r['engine'] in ENGINE_ORDER else 99)

    names = [r['engine_display'] for r in results]
    times = [r['total_time_seconds'] for r in results]
    rt_factors = [r['realtime_factor'] for r in results]
    colors = [get_color(r['engine']) for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Time
    bars1 = ax1.bar(names, times, color=colors, edgecolor='white', linewidth=1.5)
    for bar, t in zip(bars1, times):
        ax1.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.01,
                 f'{t:.3f}s', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Offline Processing Time', fontsize=14)
    ax1.set_ylim(bottom=0)

    # RT factor
    bars2 = ax2.bar(names, rt_factors, color=colors, edgecolor='white', linewidth=1.5)
    for bar, rt in zip(bars2, rt_factors):
        ax2.text(bar.get_x() + bar.get_width() / 2., bar.get_height() + 0.5,
                 f'{rt:.1f}x', ha='center', va='bottom', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Real-Time Factor', fontsize=12)
    ax2.set_title('Offline Real-Time Factor', fontsize=14)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'offline_comparison.png'), dpi=150)
    plt.close()


def generate_all_graphs(json_path, output_dir=None):
    """Generate all benchmark graphs from a JSON results file."""
    data = load_results(json_path)

    if output_dir is None:
        output_dir = os.path.dirname(json_path)

    os.makedirs(output_dir, exist_ok=True)

    print(f"Generating graphs from: {json_path}")
    print(f"Output directory: {output_dir}")

    plot_streaming_total_time(data, output_dir)
    print("  Generated: streaming_total_time.png")

    plot_per_hop_latency(data, output_dir)
    print("  Generated: per_hop_latency.png")

    plot_hop_timeline(data, output_dir)
    print("  Generated: hop_timeline.png")

    plot_realtime_factor(data, output_dir)
    print("  Generated: realtime_factor.png")

    plot_init_time(data, output_dir)
    print("  Generated: init_time.png")

    plot_offline_comparison(data, output_dir)
    print("  Generated: offline_comparison.png")

    print("Done!")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    if os.path.isdir(path):
        json_files = sorted(Path(path).glob('benchmark_*.json'))
        if not json_files:
            print(f"No benchmark JSON files found in {path}")
            sys.exit(1)
        for jf in json_files:
            generate_all_graphs(str(jf), output_dir or str(jf.parent))
    else:
        generate_all_graphs(path, output_dir)


if __name__ == '__main__':
    main()
