import torch
import torch.nn as nn
import numpy as np
import json
import os
from torchinfo import (
    summary,
)
from collections import OrderedDict
from typing import Dict
import re
from ..FxPyTorch.transparent.trans_layernorm import LayerNormTransparent


def _process_tensor(tensor: torch.Tensor, transpose: bool = False) -> Dict:
    """Helper to process a single tensor into the desired dict format."""
    if tensor is None:
        return {"values": None, "shape": None, "num_values": 0, "original": None}

    original_tensor = tensor.cpu().numpy()
    processed_tensor = original_tensor
    if transpose:
        # Ensure tensor is at least 2D for transpose
        if tensor.ndim >= 2:
            processed_tensor = np.transpose(
                original_tensor, axes=(1, 0) + tuple(range(2, tensor.ndim))
            )  # Transpose first two dims
        else:
            print(
                f"Warning: Cannot transpose tensor with shape {tensor.shape}. Skipping transpose."
            )

    return {
        "values": processed_tensor.flatten().tolist(),
        "shape": list(tensor.shape),  # Store original shape
        "num_values": tensor.numel(),
        "original": original_tensor.tolist(),  # Store original (non-transposed) values
    }


def process_linear_weights(state_dict, layer_name_prefix):
    processed = {"type": "Linear"}
    weight_key = f"{layer_name_prefix}.weight"
    bias_key = f"{layer_name_prefix}.bias"

    if weight_key not in state_dict:
        raise KeyError(
            f"Weight key '{weight_key}' not found in state_dict for Linear layer processing."
        )

    processed["weight"] = _process_tensor(state_dict[weight_key], transpose=True)

    # Bias is optional
    if bias_key in state_dict:
        processed["bias"] = _process_tensor(state_dict[bias_key], transpose=False)
    else:
        processed["bias"] = _process_tensor(None)  # Indicate bias doesn't exist

    return processed


def process_layer_norm_weights(state_dict, layer_name_prefix):
    processed = {"type": "LayerNorm"}
    weight_key = f"{layer_name_prefix}.weight"
    bias_key = f"{layer_name_prefix}.bias"

    if weight_key not in state_dict:
        # Affine might be false
        processed["weight"] = _process_tensor(None)
    else:
        processed["weight"] = _process_tensor(state_dict[weight_key], transpose=False)

    # Bias is optional even if affine is true
    if bias_key in state_dict:
        processed["bias"] = _process_tensor(state_dict[bias_key], transpose=False)
    else:
        processed["bias"] = _process_tensor(None)

    return processed


# --- New Processing Functions ---


def process_multihead_attention_weights(state_dict, layer_name_prefix):
    """
    Processes weights for MultiheadAttention layers.
    Handles both standard (in_proj) and transparent (qlinear, klinear, vlinear) formats.
    """
    processed = {"type": "MultiheadAttention"}

    # --- Check for Transparent/FxP format first ---
    q_weight_key = f"{layer_name_prefix}.qlinear.weight"
    k_weight_key = f"{layer_name_prefix}.klinear.weight"
    v_weight_key = f"{layer_name_prefix}.vlinear.weight"
    out_weight_key = f"{layer_name_prefix}.out_proj.weight"

    q_bias_key = f"{layer_name_prefix}.qlinear.bias"
    k_bias_key = f"{layer_name_prefix}.klinear.bias"
    v_bias_key = f"{layer_name_prefix}.vlinear.bias"
    out_bias_key = f"{layer_name_prefix}.out_proj.bias"

    if q_weight_key in state_dict:
        # Found transparent/FxP format
        print(
            f"Processing MHA '{layer_name_prefix}' using transparent format (qlinear, etc.)."
        )
        processed["q_proj"] = {
            "weight": _process_tensor(state_dict[q_weight_key], transpose=True),
            "bias": _process_tensor(
                state_dict.get(q_bias_key), transpose=False
            ),  # Use .get() for optional bias
        }
        processed["k_proj"] = {
            "weight": _process_tensor(state_dict[k_weight_key], transpose=True),
            "bias": _process_tensor(state_dict.get(k_bias_key), transpose=False),
        }
        processed["v_proj"] = {
            "weight": _process_tensor(state_dict[v_weight_key], transpose=True),
            "bias": _process_tensor(state_dict.get(v_bias_key), transpose=False),
        }
        processed["out_proj"] = {
            "weight": _process_tensor(state_dict[out_weight_key], transpose=True),
            "bias": _process_tensor(state_dict.get(out_bias_key), transpose=False),
        }

    else:
        # --- Check for Standard nn.MultiheadAttention format ---
        in_proj_weight_key = f"{layer_name_prefix}.in_proj_weight"
        in_proj_bias_key = f"{layer_name_prefix}.in_proj_bias"
        # Output projection keys are usually consistent
        out_weight_key = f"{layer_name_prefix}.out_proj.weight"
        out_bias_key = f"{layer_name_prefix}.out_proj.bias"

        if in_proj_weight_key in state_dict:
            print(
                f"Processing MHA '{layer_name_prefix}' using standard format (in_proj_weight)."
            )
            # Found standard format (like nn.MultiheadAttention or DPMultiheadAttention)
            in_proj_weight = state_dict[in_proj_weight_key]
            # Weights are typically stored as (out_dim * 3, in_dim)
            # We chunk along dim 0
            q_weight, k_weight, v_weight = in_proj_weight.chunk(3, dim=0)

            processed["q_proj"] = {"weight": _process_tensor(q_weight, transpose=True)}
            processed["k_proj"] = {"weight": _process_tensor(k_weight, transpose=True)}
            processed["v_proj"] = {"weight": _process_tensor(v_weight, transpose=True)}

            # Handle optional combined bias
            if in_proj_bias_key in state_dict:
                in_proj_bias = state_dict[in_proj_bias_key]
                q_bias, k_bias, v_bias = in_proj_bias.chunk(3, dim=0)
                processed["q_proj"]["bias"] = _process_tensor(q_bias, transpose=False)
                processed["k_proj"]["bias"] = _process_tensor(k_bias, transpose=False)
                processed["v_proj"]["bias"] = _process_tensor(v_bias, transpose=False)
            else:
                # No combined bias present
                processed["q_proj"]["bias"] = _process_tensor(None)
                processed["k_proj"]["bias"] = _process_tensor(None)
                processed["v_proj"]["bias"] = _process_tensor(None)

            # Process output projection (should exist in both formats)
            if out_weight_key not in state_dict:
                raise KeyError(
                    f"Output projection weight key '{out_weight_key}' not found for standard MHA processing."
                )
            processed["out_proj"] = {
                "weight": _process_tensor(state_dict[out_weight_key], transpose=True),
                "bias": _process_tensor(
                    state_dict.get(out_bias_key), transpose=False
                ),  # Optional bias
            }
        else:
            raise KeyError(
                f"Could not determine MHA format for prefix '{layer_name_prefix}'. "
                f"Neither '{q_weight_key}' nor '{in_proj_weight_key}' found in state_dict."
            )

    return processed


def process_transformer_encoder_layer(state_dict, layer_name_prefix):
    """
    Processes weights for a single TransformerEncoderLayer and its sub-components.
    """
    processed = {"type": "TransformerEncoderLayer"}
    print(f"Processing TransformerEncoderLayer: {layer_name_prefix}")

    # Process Self-Attention
    attn_prefix = f"{layer_name_prefix}.self_attn"
    # Check if self_attn weights exist before processing
    if any(k.startswith(attn_prefix) for k in state_dict.keys()):
        processed["self_attn"] = process_multihead_attention_weights(
            state_dict, attn_prefix
        )
    else:
        print(
            f"Warning: No self_attn weights found for prefix '{attn_prefix}'. Skipping."
        )
        processed["self_attn"] = {
            "type": "MultiheadAttention",
            "error": "Weights not found",
        }

    # Process Feed-Forward Linear Layers
    linear1_prefix = f"{layer_name_prefix}.linear1"
    if any(k.startswith(linear1_prefix) for k in state_dict.keys()):
        processed["linear1"] = process_linear_weights(state_dict, linear1_prefix)
    else:
        print(
            f"Warning: No linear1 weights found for prefix '{linear1_prefix}'. Skipping."
        )
        processed["linear1"] = {"type": "Linear", "error": "Weights not found"}

    linear2_prefix = f"{layer_name_prefix}.linear2"
    if any(k.startswith(linear2_prefix) for k in state_dict.keys()):
        processed["linear2"] = process_linear_weights(state_dict, linear2_prefix)
    else:
        print(
            f"Warning: No linear2 weights found for prefix '{linear2_prefix}'. Skipping."
        )
        processed["linear2"] = {"type": "Linear", "error": "Weights not found"}

    # Process Layer Normalization Layers
    norm1_prefix = f"{layer_name_prefix}.norm1"
    if any(k.startswith(norm1_prefix) for k in state_dict.keys()):
        processed["norm1"] = process_layer_norm_weights(state_dict, norm1_prefix)
    else:
        print(f"Warning: No norm1 weights found for prefix '{norm1_prefix}'. Skipping.")
        processed["norm1"] = {"type": "LayerNorm", "error": "Weights not found"}

    norm2_prefix = f"{layer_name_prefix}.norm2"
    if any(k.startswith(norm2_prefix) for k in state_dict.keys()):
        processed["norm2"] = process_layer_norm_weights(state_dict, norm2_prefix)
    else:
        print(f"Warning: No norm2 weights found for prefix '{norm2_prefix}'. Skipping.")
        processed["norm2"] = {"type": "LayerNorm", "error": "Weights not found"}

    return processed


# --- Main Conversion Function for Attention Autoencoder ---


def convert_model_to_json_ae(model, filename="model_weights_ae.json", input_shape=None):
    """
    Converts an AttentionAutoencoderTransparent or FxpAttentionAutoencoder model's weights
    to a structured JSON file.
    Tested and working only on Transparent implementation
    """
    # --- Get Model Summary ---
    summary_str = "torchinfo summary not available."
    if input_shape is not None:
        try:
            # Use a large depth to capture nested layers
            summary_str = str(
                summary(model, input_shape=input_shape, depth=5, verbose=0)
            )
            print("Generated model summary.")
        except Exception as e:
            summary_str = f"torchinfo summary failed: {e}"
            print(f"Warning: torchinfo summary failed: {e}")
    else:
        print("Warning: input_shape not provided, skipping torchinfo summary.")

    # --- Get State Dict ---
    state_dict = model.state_dict()
    print(f"Extracted state_dict with {len(state_dict)} entries.")

    # --- Initialize Output Dictionary ---
    processed_dict = OrderedDict()  # Use OrderedDict to maintain some order
    processed_dict["model_summary"] = summary_str
    processed_dict["model_architecture"] = str(model)
    processed_dict["weights"] = OrderedDict()  # Store weights under a specific key

    # --- Identify and Process Layers using named_modules ---
    # This is generally more robust than parsing keys from state_dict directly
    processed_prefixes = (
        set()
    )  # Keep track of processed prefixes to avoid double processing

    print("\nProcessing model layers...")
    for name, module in model.named_modules():
        if not name:  # Skip root module
            continue

        # Check if this module's prefix has already been handled by a parent processor
        is_processed = False
        for prefix in processed_prefixes:
            if name.startswith(prefix + "."):
                is_processed = True
                break
        if is_processed:
            continue

        print(f"Checking module: {name} (Type: {type(module).__name__})")

        processed_layer = None
        processed_prefix = name  # Assume we process this exact module name

        # --- Layer Type Identification and Processing ---
        if isinstance(module, (nn.Linear)):
            # Basic Linear Layer (covers embed, compress.0, output, and FFN linears if not inside EncoderLayer)
            try:
                processed_layer = process_linear_weights(state_dict, name)
            except KeyError as e:
                print(f"  Skipping Linear layer '{name}': {e}")
            except Exception as e:
                print(f"  Error processing Linear layer '{name}': {e}")
                processed_layer = {"type": "Linear", "error": str(e)}

        elif isinstance(module, (nn.LayerNorm, LayerNormTransparent)):
            # Basic LayerNorm (covers transformer_encoder.norm or norm_layer)
            try:
                processed_layer = process_layer_norm_weights(state_dict, name)
            except KeyError as e:
                print(f"  Skipping LayerNorm layer '{name}': {e}")
            except Exception as e:
                print(f"  Error processing LayerNorm layer '{name}': {e}")
                processed_layer = {"type": "LayerNorm", "error": str(e)}

        elif isinstance(module, (nn.MultiheadAttention)):
            # Standalone MHA (unlikely in AE structure, but for completeness)
            # Note: DPMultiheadAttention inherits from nn.MultiheadAttention
            try:
                processed_layer = process_multihead_attention_weights(state_dict, name)
            except KeyError as e:
                print(f"  Skipping MultiheadAttention layer '{name}': {e}")
            except Exception as e:
                print(f"  Error processing MultiheadAttention layer '{name}': {e}")
                processed_layer = {"type": "MultiheadAttention", "error": str(e)}

        elif isinstance(module, (nn.TransformerEncoderLayer)):
            # Process the entire encoder layer block
            try:
                processed_layer = process_transformer_encoder_layer(state_dict, name)
                processed_prefix = name  # Mark this whole block as processed
            except KeyError as e:
                print(f"  Skipping TransformerEncoderLayer '{name}': {e}")
            except Exception as e:
                print(f"  Error processing TransformerEncoderLayer '{name}': {e}")
                processed_layer = {"type": "TransformerEncoderLayer", "error": str(e)}

        # --- Add other layer types if needed ---
        elif isinstance(module, nn.ModuleList) and name == "transformer_encoder.layers":
            # Special handling for the standard nn.TransformerEncoder's ModuleList
            print(f"  Found standard TransformerEncoder layers container: {name}")
            # Process each layer inside the list
            for i, sub_module in enumerate(module):
                sub_layer_name = f"{name}.{i}"
                if isinstance(sub_module, (nn.TransformerEncoderLayer)):
                    try:
                        layer_data = process_transformer_encoder_layer(
                            state_dict, sub_layer_name
                        )
                        processed_dict["weights"][sub_layer_name] = layer_data
                        processed_prefixes.add(sub_layer_name)  # Mark as processed
                        print(f"    Processed {sub_layer_name}")
                    except Exception as e:
                        print(f"    Error processing {sub_layer_name}: {e}")
                        processed_dict["weights"][sub_layer_name] = {
                            "type": "TransformerEncoderLayer",
                            "error": str(e),
                        }
                else:
                    print(
                        f"    Skipping non-encoder layer within {name}: {type(sub_module).__name__}"
                    )
            continue  # Skip adding the ModuleList itself

        elif isinstance(module, (nn.Dropout, nn.ReLU, nn.Sequential)):
            print(
                f"  Skipping layer with no weights: {name} (Type: {type(module).__name__})"
            )
            continue  # Skip layers without weights or containers handled elsewhere

        elif "pos_enc" in name:
            print(f"  Skipping PositionalEncoding layer: {name}")
            continue  # Positional encoding uses buffers, not typical weights/biases

        else:
            # Check if it has parameters directly (e.g., custom layers not caught above)
            # This check might be redundant if state_dict keys are handled correctly
            has_params = any(p.requires_grad for p in module.parameters(recurse=False))
            has_buffers = len(list(module.buffers(recurse=False))) > 0
            if has_params or has_buffers:
                print(
                    f"  Potentially unhandled layer with parameters/buffers: {name} (Type: {type(module).__name__}). Check state_dict keys manually if needed."
                )
            else:
                print(
                    f"  Skipping non-parameter layer/container: {name} (Type: {type(module).__name__})"
                )
            continue

        # --- Store Processed Layer Data ---
        if processed_layer:
            processed_dict["weights"][name] = processed_layer
            processed_prefixes.add(processed_prefix)  # Add the processed prefix
            print(f"  Processed: {name}")

    # --- Save to JSON ---
    print(f"\nSaving processed weights to {filename}...")
    try:
        with open(filename, "w") as f:
            json.dump(
                processed_dict, f, indent=4, cls=NpEncoder
            )  # Use custom encoder for numpy types if any remain
        print("Successfully saved JSON file.")
    except Exception as e:
        print(f"Error saving JSON file: {e}")


def write_array_to_file(
    file, name, values, num_values, shape, data_type, beautify: bool = True
):
    """
    Write the C array representation of the weights and biases to the file
    with original shape as a comment. Handles None values.
    """
    if values is None or num_values == 0:
        file.write(
            f"// const {data_type} {name}[0] = {{}}; // Layer did not contain this parameter\n\n"
        )
        return

    # Ensure shape is valid before formatting
    if shape is None:
        shape_str = "N/A"
    else:
        try:
            shape_str = " x ".join(
                map(str, shape)
            )  # Convert shape to a string format "x1 x x2"
        except TypeError:
            shape_str = "Invalid Shape"

    file.write(
        f"const {data_type} {name}[{num_values}] = {{  // Original shape: {shape_str}\n"
    )
    for i, value in enumerate(values):
        # Ensure value is a number before formatting
        if isinstance(value, (int, float)):
            formatted_value = f"{value:.9f}f"
        else:
            formatted_value = "/* Invalid Value */"  # Placeholder for non-numeric data

        end_char = "," if i < num_values - 1 else ""  # No comma after the last element
        file.write(f"    {formatted_value}{end_char}")
        if beautify and (i + 1) % 8 == 0 and i < num_values - 1:
            file.write("\n")
    file.write("\n};\n\n")


def write_linear_layer_to_h(
    layer_name, layer_data, file, weight_type="weight_t", bias_type="bias_t"
):
    """Process Linear layers and write their arrays to the file."""
    formatted_name = layer_name.replace(".", "_")
    weights_data = layer_data.get("weight", {})  # Use .get() for safety
    bias_data = layer_data.get("bias", {})

    print(f"  Writing Linear Layer: {layer_name} -> {formatted_name}")
    write_array_to_file(
        file,
        f"linear_{formatted_name}_weights",
        weights_data.get("values"),
        weights_data.get("num_values", 0),
        weights_data.get("shape"),
        weight_type,
    )
    write_array_to_file(
        file,
        f"linear_{formatted_name}_bias",
        bias_data.get("values"),
        bias_data.get("num_values", 0),
        bias_data.get("shape"),
        bias_type,
    )


def write_layernorm_layer_to_h(
    layer_name, layer_data, file, weight_type="weight_t", bias_type="bias_t"
):
    """Process LayerNorm layers and write their arrays to the file."""
    formatted_name = layer_name.replace(".", "_")
    weights_data = layer_data.get("weight", {})
    bias_data = layer_data.get("bias", {})

    print(f"  Writing LayerNorm Layer: {layer_name} -> {formatted_name}")
    write_array_to_file(
        file,
        f"layernorm_{formatted_name}_weights",
        weights_data.get("values"),
        weights_data.get("num_values", 0),
        weights_data.get("shape"),
        weight_type,
    )
    write_array_to_file(
        file,
        f"layernorm_{formatted_name}_bias",
        bias_data.get("values"),
        bias_data.get("num_values", 0),
        bias_data.get("shape"),
        bias_type,
    )


# --- New Function to Write MultiheadAttention Layer ---


def write_mha_layer_to_h(
    layer_name, layer_data, file, weight_type="weight_t", bias_type="bias_t"
):
    """Process MultiheadAttention layers and write their arrays to the file."""
    formatted_name = layer_name.replace(".", "_")
    print(f"  Writing MultiheadAttention Layer: {layer_name} -> {formatted_name}")

    projections = ["q_proj", "k_proj", "v_proj", "out_proj"]
    for proj_name in projections:
        proj_data = layer_data.get(proj_name)
        if not proj_data:
            print(f"    Skipping {proj_name} for {layer_name}: Data not found in JSON.")
            continue

        weights_data = proj_data.get("weight", {})
        bias_data = proj_data.get("bias", {})
        c_proj_name = proj_name.replace("_proj", "")  # e.g., "q", "k", "v", "out"

        # Write Weight
        write_array_to_file(
            file,
            f"mha_{formatted_name}_{c_proj_name}_weights",
            weights_data.get("values"),
            weights_data.get("num_values", 0),
            weights_data.get("shape"),
            weight_type,
        )
        # Write Bias
        write_array_to_file(
            file,
            f"mha_{formatted_name}_{c_proj_name}_bias",
            bias_data.get("values"),
            bias_data.get("num_values", 0),
            bias_data.get("shape"),
            bias_type,
        )


# --- Main Conversion Function for Attention Autoencoder ---


def write_transformer_encoder_layer_to_h(
    layer_name, layer_data, file, weight_type="model_weight_t", bias_type="model_bias_t"
):
    """
    Processes the nested structure of a TransformerEncoderLayer from the JSON
    and calls the appropriate writers for its sub-components.
    """
    formatted_name = layer_name.replace(".", "_")
    print(f"Writing TransformerEncoderLayer: {layer_name} -> {formatted_name}")
    file.write(f"// --- Weights for Transformer Encoder Layer: {layer_name} ---\n")

    # Define the expected sub-modules within the encoder layer data
    sub_modules = {
        "self_attn": ("multiheadattention", write_mha_layer_to_h),
        "linear1": ("linear", write_linear_layer_to_h),
        "linear2": ("linear", write_linear_layer_to_h),
        "norm1": ("layernorm", write_layernorm_layer_to_h),
        "norm2": ("layernorm", write_layernorm_layer_to_h),
    }

    for sub_name, (expected_type, writer_func) in sub_modules.items():
        sub_data = layer_data.get(sub_name)
        full_sub_name = f"{layer_name}.{sub_name}"  # Construct the full path name

        if not sub_data:
            print(
                f"  Warning: Sub-module '{sub_name}' not found in data for '{layer_name}'. Skipping."
            )
            file.write(
                f"// Sub-module {full_sub_name.replace('.', '_')} not found in JSON for {formatted_name}.\n\n"
            )
            continue

        if "error" in sub_data:
            print(
                f"  Warning: Skipping sub-module '{sub_name}' for '{layer_name}' due to error: {sub_data['error']}"
            )
            file.write(
                f"// Skipped sub-module {full_sub_name.replace('.', '_')} for {formatted_name} due to error: {sub_data['error']}\n\n"
            )
            continue

        actual_type = sub_data.get("type", "").lower()
        if actual_type != expected_type:
            print(
                f"  Warning: Expected type '{expected_type}' for sub-module '{sub_name}' in '{layer_name}', but found '{actual_type}'. Skipping."
            )
            file.write(
                f"// Skipped sub-module {full_sub_name.replace('.', '_')} for {formatted_name} - Unexpected type: {actual_type}.\n\n"
            )
            continue

        # Call the specific writer function for this sub-module
        writer_func(full_sub_name, sub_data, file, weight_type, bias_type)

    file.write(
        f"// --- End Weights for Transformer Encoder Layer: {layer_name} ---\n\n"
    )


def convert_json_to_h_ae(
    json_filename,
    h_filename,
    include_path='"../parameters.h"',  # Default include path for types
    header_guard_define="MODEL_AE_WEIGHTS_H_",
    weight_type="weight_t",  # C type for weights
    bias_type="bias_t",
):  # C type for biases
    """
    Convert the JSON model file (generated by convert_model_to_json_ae)
    to a C header file with arrays for weights and biases.
    """
    print(f"Attempting to load JSON data from: {json_filename}")
    try:
        with open(json_filename, "r") as json_file:
            model_data = json.load(json_file)
        print("JSON data loaded successfully.")
    except FileNotFoundError:
        print(f"Error: JSON file not found at {json_filename}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from {json_filename}. Details: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred while loading JSON: {e}")
        return

    # Extract the weights dictionary
    weights_dict = model_data.get("weights")
    if not weights_dict:
        print("Error: 'weights' key not found or is empty in the JSON data.")
        return
    if not isinstance(weights_dict, dict):
        print(
            f"Error: Expected 'weights' to be a dictionary, but got {type(weights_dict)}."
        )
        return

    print(
        f"Processing {len(weights_dict)} layer entries from the 'weights' dictionary."
    )

    try:
        with open(h_filename, "w") as h_file:
            # Add the include and header guard lines
            h_file.write(f"#include {include_path}\n\n")  # Include file defining types
            h_file.write(f"#ifndef {header_guard_define}\n")
            h_file.write(f"#define {header_guard_define}\n\n")
            h_file.write(
                f"// Auto-generated weights from {os.path.basename(json_filename)}\n\n"
            )

            # --- Process each layer in the weights dictionary ---
            for layer_name, layer_data in weights_dict.items():
                if not isinstance(layer_data, dict):
                    print(
                        f"Warning: Skipping entry '{layer_name}' - expected a dictionary, got {type(layer_data)}."
                    )
                    continue

                layer_type = layer_data.get("type", "").lower()
                if not layer_type:
                    print(
                        f"Warning: Skipping entry '{layer_name}' - 'type' key not found."
                    )
                    continue

                if "error" in layer_data:
                    print(
                        f"Warning: Skipping entry '{layer_name}' due to processing error in JSON: {layer_data['error']}"
                    )
                    h_file.write(
                        f"// Skipped layer {layer_name.replace('.', '_')} due to JSON processing error: {layer_data['error']}\n\n"
                    )
                    continue

                # --- Call appropriate writer function based on type ---
                if layer_type == "linear":
                    write_linear_layer_to_h(
                        layer_name, layer_data, h_file, weight_type, bias_type
                    )
                elif layer_type == "layernorm":
                    write_layernorm_layer_to_h(
                        layer_name, layer_data, h_file, weight_type, bias_type
                    )
                elif layer_type == "multiheadattention":
                    write_mha_layer_to_h(
                        layer_name, layer_data, h_file, weight_type, bias_type
                    )
                elif layer_type == "transformerencoderlayer":
                    write_transformer_encoder_layer_to_h(
                        layer_name, layer_data, h_file, weight_type, bias_type
                    )
                # Add elif for other layer types if needed in the future
                # elif layer_type == 'lstm': # Example if LSTM was used
                #     write_lstm_layer_to_h(layer_name, layer_data, h_file)
                else:
                    print(
                        f"Warning: Unrecognized layer type '{layer_type}' for layer '{layer_name}'. Skipping C code generation for this layer."
                    )
                    h_file.write(
                        f"// Skipped layer {layer_name.replace('.', '_')} - Unrecognized type: {layer_type}\n\n"
                    )

            # Add the end of the header guard
            h_file.write(f"#endif // {header_guard_define}\n")

        print(f"\nHeader file '{h_filename}' created successfully.")

    except IOError as e:
        print(f"Error: Failed to write to header file {h_filename}. Details: {e}")
    except Exception as e:
        print(f"An unexpected error occurred during C header generation: {e}")


# Custom JSON encoder to handle potential numpy types if direct conversion fails somewhere
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def _get_cpp_type_prefix(variable_name: str) -> str:
    """
    Maps a C variable name to a specific C++ type prefix based on naming conventions.
    This version is specifically tailored to the observed variable name patterns.
    """
    # Order is important: more specific rules must come first.

    # --- Multi-Head Attention sub-layers ---
    # These are the most specific, so we check for them first.
    # The key is to check for 'self_attn' AND the projection name ('qlinear', etc.).
    if "self_attn" in variable_name and "qlinear" in variable_name:
        return "mha_q"
    if "self_attn" in variable_name and "klinear" in variable_name:
        return "mha_k"
    if "self_attn" in variable_name and "vlinear" in variable_name:
        return "mha_v"
    if "self_attn" in variable_name and "out_proj" in variable_name:
        return "mha_out"

    # --- Transformer Feed-Forward and Norm layers ---
    # These rules were working correctly but are kept for completeness.
    if "linear1" in variable_name:
        return "ff1"
    if "linear2" in variable_name:
        return "ff2"
    if "norm1" in variable_name:
        return "ln1"
    if "norm2" in variable_name:
        return "ln2"

    # --- Top-level/Outer Layers ---
    if "linear_embed" in variable_name:
        return "l_embed"
    # Use a more specific key to avoid accidental matches
    if "compress_0" in variable_name:
        return "l_compress"
    if "linear_output" in variable_name:
        return "l_out"
    # This is the final LayerNorm outside the encoder block
    if "layernorm_norm_layer" in variable_name:
        return "ln3"

    # If no rule matches, raise an error to make it obvious.
    raise ValueError(f"No C++ type mapping rule found for variable: '{variable_name}'")


def post_process_header_for_specific_types(h_filename: str):
    """
    Reads a generated C header file and replaces generic types (weight_t, bias_t)
    with specific C++ types (e.g., l_embed_weight_t, mha_q_accum_t) based on
    the variable names.

    This function modifies the file in-place.

    Args:
        h_filename (str): The path to the header file to process.
    """
    print(f"Post-processing header for specific C++ types: {h_filename}")
    try:
        with open(h_filename, "r") as f:
            lines = f.readlines()

        processed_lines = []
        # Regex to capture the type, variable name, and the rest of the line
        pattern = re.compile(
            r"^\s*const\s+(weight_t|bias_t)\s+([\w_]+)(\s*\[.*\]\s*=.*)"
        )

        for line in lines:
            match = pattern.match(line)
            if match:
                original_type = match.group(1)
                variable_name = match.group(2)
                rest_of_line = match.group(3)

                try:
                    type_prefix = _get_cpp_type_prefix(variable_name)

                    if original_type == "weight_t":
                        new_type = f"{type_prefix}_weight_t"
                    else:  # bias_t
                        new_type = f"{type_prefix}_accum_t"

                    # Reconstruct the line with the new type
                    new_line = f"const {new_type} {variable_name}{rest_of_line}\n"
                    processed_lines.append(new_line)
                    print(f"  - Mapped '{variable_name}' -> {new_type}")

                except ValueError as e:
                    print(f"  - WARNING: {e}. Keeping original line.")
                    processed_lines.append(line)  # Keep original if no rule found
            else:
                processed_lines.append(line)  # Keep non-matching lines as-is

        # Write the processed lines back to the file
        with open(h_filename, "w") as f:
            f.writelines(processed_lines)

        print("Header post-processing complete.")

    except FileNotFoundError:
        print(f"Error: Could not find header file to post-process: {h_filename}")
    except Exception as e:
        print(f"An unexpected error occurred during header post-processing: {e}")
