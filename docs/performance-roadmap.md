# DeepFilterNet-mlx Performance Roadmap

Goal: beat current Rust DeepFilterNet runtime in both offline and low-latency streaming modes.

## Current architecture

- Swift MLX model forward with stateful streaming runtime
- Custom MLXFast fused kernels for:
  - ERB mask projection multiply (`spec * gains`)
  - Streaming deep-filter complex accumulation per frame

## Next optimization phases

1. End-to-end compiled graph for fixed-shape streaming hop
- Compile one-hop streaming path with `compile(shapeless: false)`
- Warmup once and keep graph hot
- Remove residual dynamic shape branches from hop path

2. Feature extraction fusion
- Fuse ERB energy + log + EMA update into one custom kernel
- Fuse DF magnitude + EMA + normalization into one custom kernel
- Keep feature state in preallocated contiguous buffers

3. Recurrent path packing
- Pack GRU gate matmuls to reduce launches
- Pretranspose and align all GRU/linear weights for contiguous GEMM
- Keep hidden states in fixed device buffers (no per-hop allocs)

4. Deep-filter stage fusion
- Full fused kernel for DF order loop + low/high bin merge
- Add optional fp16 path with fp32 accumulator for quality/speed balance

5. IO + scheduling
- Double-buffered producer/consumer for audio chunk ingest
- Isolate file IO from inference critical path
- Pin chunk size to one hop for minimum command-buffer latency

## Benchmark protocol

Use two workloads:

- Offline: 10s / 60s / 300s clips
- Streaming: 10ms and 20ms chunking, real-time factor and p95 latency

Metrics:

- RTF (real-time factor)
- p50/p95/p99 per-hop latency
- GPU kernel launch count per hop
- Peak memory

Baseline comparisons:

- Rust `deep-filter` offline
- Rust streaming binary path with matching delay compensation/padding
- Python MLX runtime for parity checks

## Quality guardrails

- Waveform SNR, SI-SDR, DNSMOS where possible
- Spectrogram delta checks versus Python MLX reference
- Streaming/offline consistency checks on same input

## Immediate priorities

1. Add benchmark harness with repeatable command lines and CSV output
2. Move streamer hop path to compiled graph
3. Fuse feature normalization kernels
4. Validate quality parity after each fusion change
