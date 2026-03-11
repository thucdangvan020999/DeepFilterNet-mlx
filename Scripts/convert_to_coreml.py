#!/usr/bin/env python3
"""
Convert DeepFilterNet3 safetensors weights to a CoreML per-hop inference model.

This script builds a CoreML model that handles one streaming hop:
  Input:  feat_erb [1, 3, 1, nbErb], feat_df [1, 3, 2, nbDf], spec [1, 1, freqBins, 2]
  Output: spec_enhanced [freqBins, 2]

The STFT/ISTFT and feature extraction remain in Swift.

Usage:
    python Scripts/convert_to_coreml.py --model-dir /path/to/DeepFilterNet3
    python Scripts/convert_to_coreml.py --model-dir /path/to/DeepFilterNet3 --output DeepFilterNet3_hop.mlpackage

Requirements:
    pip install coremltools safetensors numpy
"""

import argparse
import json
import os
import sys

try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    from coremltools.converters.mil.mil import types
    import numpy as np
    from safetensors import safe_open
except ImportError as e:
    print(f"Missing dependency: {e}")
    print("Install with: pip install coremltools safetensors numpy")
    sys.exit(1)


def load_config(model_dir):
    """Load model config.json."""
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path) as f:
        return json.load(f)


def load_weights(model_dir):
    """Load safetensors weights."""
    weights = {}
    for fname in os.listdir(model_dir):
        if fname.endswith('.safetensors'):
            path = os.path.join(model_dir, fname)
            with safe_open(path, framework="numpy") as f:
                for key in f.keys():
                    weights[key] = f.get_tensor(key)
    return weights


def build_coreml_model(config, weights):
    """Build CoreML model using coremltools MIL builder."""
    nb_erb = config.get("nb_erb", 32)
    nb_df = config.get("nb_df", 96)
    freq_bins = config.get("fft_size", 960) // 2 + 1
    df_order = config.get("df_order", 5)
    conv_ch = config.get("conv_ch", 64)
    emb_hidden_dim = config.get("emb_hidden_dim", 256)

    print(f"Building CoreML model:")
    print(f"  nb_erb={nb_erb}, nb_df={nb_df}, freq_bins={freq_bins}")
    print(f"  df_order={df_order}, conv_ch={conv_ch}, emb_hidden={emb_hidden_dim}")

    # For now, create a simple passthrough model as a placeholder.
    # A full implementation would recreate the entire encoder-decoder architecture
    # using MIL operations, which requires significant effort.
    #
    # The key operations to implement:
    # 1. Conv2d blocks (encoder, with BN fusion)
    # 2. GRU cells (3 separate: enc_emb, erb_dec, df_dec)
    # 3. ConvTranspose2d (decoder)
    # 4. Grouped linear layers
    # 5. Deep filter application
    #
    # This is a multi-hundred-line MIL program. For the benchmark,
    # the Swift CoreML engine handles the case where no model exists
    # by outputting silence, measuring just the overhead.

    @mb.program(
        input_specs=[
            mb.TensorSpec(shape=(1, 1, freq_bins, 2), dtype=types.fp32),  # spec
            mb.TensorSpec(shape=(1, 3, 1, nb_erb), dtype=types.fp32),     # feat_erb
            mb.TensorSpec(shape=(1, 3, 2, nb_df), dtype=types.fp32),      # feat_df
        ]
    )
    def prog(spec, feat_erb, feat_df):
        # Placeholder: passthrough the input spectrum
        # In a full implementation, this would be the entire encoder-decoder network
        out = mb.squeeze(x=spec, axes=[0, 1])  # [freq_bins, 2]
        return out

    # Convert to CoreML
    model = ct.convert(
        prog,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS14,
        compute_precision=ct.precision.FLOAT32,
    )

    # Set metadata
    model.author = "DeepFilterNet-mlx benchmark"
    model.short_description = "DeepFilterNet3 per-hop streaming inference (placeholder)"
    model.input_description["spec"] = "Input spectrum [1, 1, freqBins, 2]"
    model.input_description["feat_erb"] = "ERB features [1, 3, 1, nbErb]"
    model.input_description["feat_df"] = "DF features [1, 3, 2, nbDf]"

    return model


def main():
    parser = argparse.ArgumentParser(description="Convert DeepFilterNet to CoreML")
    parser.add_argument("--model-dir", required=True, help="Path to model directory with config.json and .safetensors")
    parser.add_argument("--output", default=None, help="Output .mlpackage path (default: <model-dir>/DeepFilterNet3_hop.mlpackage)")
    args = parser.parse_args()

    if not os.path.isdir(args.model_dir):
        print(f"Error: {args.model_dir} is not a directory")
        sys.exit(1)

    config = load_config(args.model_dir)
    weights = load_weights(args.model_dir)
    print(f"Loaded {len(weights)} weight tensors")

    model = build_coreml_model(config, weights)

    output_path = args.output or os.path.join(args.model_dir, "DeepFilterNet3_hop.mlpackage")
    model.save(output_path)
    print(f"Saved CoreML model to: {output_path}")
    print("Note: This is a placeholder model. Full architecture conversion is needed for accurate benchmarks.")


if __name__ == "__main__":
    main()
