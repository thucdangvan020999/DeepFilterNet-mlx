# DeepFilterNet-mlx

Standalone Swift/MLX implementation of DeepFilterNet with a performance-first runtime for Apple Silicon.

This repo was split out from `mlx-audio-swift` so DeepFilterNet can evolve independently with fused kernels, custom Metal paths, and streaming-latency optimizations.

## What is implemented

- Full Swift MLX DeepFilterNet runtime (`DeepFilterNet`, `DeepFilterNet2`, `DeepFilterNet3` checkpoints)
- Offline enhancement (`enhance(_:)`)
- Stateful streaming enhancement (`DeepFilterNetStreamer`)
- Local/Hugging Face model loading (`config.json` + `.safetensors`)
- Standalone DSP layer (STFT/ISTFT) without `mlx-audio-swift` dependency
- Optional fused Metal kernels via `MLXFast`:
  - ERB mask multiply fusion
  - Streaming deep-filter frame fusion

## Repo layout

- `Sources/DeepFilterNetMLX`: library code
- `Sources/deepfilternet-mlx-cli`: CLI tool
- `Scripts/convert_deepfilternet.py`: PyTorch-to-MLX weight conversion
- `Tests/DeepFilterNetMLXTests`: basic smoke tests
- `docs/performance-roadmap.md`: plan to beat Rust latency/throughput

## Build

```bash
swift build
```

## Test

```bash
swift test
```

## CLI usage

```bash
swift run deepfilternet-mlx /path/to/noisy.wav \
  --model /path/to/DeepFilterNet3 \
  --output /path/to/enhanced.wav
```

Streaming mode (10 ms chunks):

```bash
swift run deepfilternet-mlx /path/to/noisy.wav \
  --model /path/to/DeepFilterNet3 \
  --stream \
  --chunk-ms 10 \
  --output /path/to/enhanced_stream.wav
```

Performance preset:

```bash
# throughput (default): enables fused kernels
--performance throughput

# safe: disables fused kernels
--performance safe
```

## Library usage

```swift
import DeepFilterNetMLX
import MLX

let model = try await DeepFilterNetModel.fromPretrained("/path/to/DeepFilterNet3")
model.configurePerformance(.throughput)

let enhanced = try model.enhance(inputAudio)
```

Or file-based helper:

```swift
try model.enhanceFile(
    inputURL: inputURL,
    outputURL: outputURL,
    useStreaming: true,
    chunkMilliseconds: 10.0
)
```

## Model conversion

```bash
python Scripts/convert_deepfilternet.py \
  --input /path/to/DeepFilterNet/checkpoint_dir \
  --output /path/to/DeepFilterNet3 \
  --name DeepFilterNet3
```

## Notes

- The package targets `macOS 14+` and `iOS 17+`.
- For best runtime performance, warm the model once before measuring.
- The current fused kernels are focused on streaming hotspots and intended as a base for deeper kernel fusion work.
