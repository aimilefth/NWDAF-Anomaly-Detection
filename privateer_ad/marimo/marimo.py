# privateer_ad/marimo/marimo.py

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import logging
    from pathlib import Path

    import numpy as np
    import os

    from privateer_ad.config import DataConfig, PathConfig, MetadataConfig
    from privateer_ad.old_data_utils import OldDataProcessor
    from privateer_ad.alveo.alveo_runner import AlveoRunner, AlveoRunnerParameters
    from privateer_ad.evaluate.evaluator_alveo import AlveoEvaluator
    import torch

    from privateer_ad.architectures import TransformerAD
    from privateer_ad.config import ModelConfig
    from privateer_ad.evaluate.evaluator import ModelEvaluator
    from privateer_ad.utils import load_model_weights

    from privateer_ad.approximation.transformer_ad_fxp import (
        FxpTransformerAD,
        FxPTransformerADConfig,
        TransformerADQConfig,
        create_dynamic_qconfig,
    )

    from privateer_ad.approximation.approximation_comparison import approximation_comparison

    logging.basicConfig(level=logging.INFO)
    mo.md("### 1) Imports ready")

    USE_OLD_METADATA = False
    PATHS = PathConfig()
    THRESHOLD = 0.0209596287459135
    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return (
        AlveoRunner,
        AlveoRunnerParameters,
        DataConfig,
        FxPTransformerADConfig,
        FxpTransformerAD,
        ModelConfig,
        OldDataProcessor,
        PATHS,
        THRESHOLD,
        TransformerAD,
        USE_OLD_METADATA,
        approximation_comparison,
        create_dynamic_qconfig,
        cuda_device,
        logging,
        mo,
        os,
        torch,
    )


@app.cell
def _(DataConfig, OldDataProcessor, PATHS, USE_OLD_METADATA, logging, mo):
    mo.md("### 2) Build data pipeline (test)")
    dconf = DataConfig(seq_len=12)  # override here if you want: DataConfig(seq_len=..., batch_size=...)
    # 1) Initialize old data processor
    dp = OldDataProcessor(use_old_metadata=USE_OLD_METADATA, recompute_attack_from_metadata=USE_OLD_METADATA)    # 2) Ensure processed splits exist; if not, build them and fit scalers
    train_csv = PATHS.processed_dir.joinpath("train.csv")
    val_csv = PATHS.processed_dir.joinpath("val.csv")
    test_csv = PATHS.processed_dir.joinpath("test.csv")
    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        # This will split per device (stratified) and fit scaler (and PCA if you pass n_components)
        dp.initialize_data_pipeline(dataset_path=PATHS.raw_dataset, train_size=0.8)

    # 3) Build a test dataloader with old defaults (seq_len=12, batch_size=4096)
    test_dl= dp.get_dataloader(
        split="test",
        use_pca=False,        # flip to True if you fitted PCA in initialize_data_pipeline(n_components=...)
        batch_size=4096,
        seq_len=12,
        partition_id=None,    # or an int if you want a federated-like partition
        only_benign=False,
        num_workers=16,
    )

    val_dl= dp.get_dataloader(
        split="val",
        use_pca=False,        # flip to True if you fitted PCA in initialize_data_pipeline(n_components=...)
        batch_size=131072,
        seq_len=12,
        partition_id=None,    # or an int if you want a federated-like partition
        only_benign=True,
        num_workers=16,
    )


    logging.info(
        f"Data ready: seq_len={dconf.seq_len}, batch_size={dconf.batch_size}, features={len(dp.input_features)}"
    )
    (len(val_dl), len(test_dl)) # Display tuple of lengths
    return dconf, dp, test_dl, val_dl


@app.cell
def _(mo):
    mo.md("### 3) Check Alveo devices (pynq)")
    try:
        import pynq
        device_list = list(pynq.Device.devices)
        for i, dev in enumerate(device_list):
            print(f"{i}) {dev.name}")
    except Exception as e:
        device_list = []
        print(f"Could not enumerate PYNQ devices. Reason: {e}")
        print("If you're not on the FPGA machine yet, this is expected.")
    device_list
    return (device_list,)


@app.cell
def _(
    AlveoRunner,
    AlveoRunnerParameters,
    PATHS,
    dconf,
    device_list,
    dp,
    mo,
    os,
):
    mo.md("### 4) Create Alveo runner")
    XCLBIN_PATH = os.path.join(PATHS.experiments_dir, "alveo_xclbins", "attention_ae_adv_W16A16.xclbin")

    # Bus can be None if you only want overlay + kernel (no power scraping)
    DEVICE_BUS = None  # e.g. "0000:af:00.1"
    DEVICE_INDEX = 0 if device_list else 0  # default 0 if available

    n_features = len(dp.input_features)
    params = AlveoRunnerParameters(
        input_buffer_elements=dconf.seq_len * n_features,
        output_buffer_elements=dconf.seq_len * n_features,
        kernel_name=None,  # set if your overlay has a named IP block
    )

    device = device_list[DEVICE_INDEX] if device_list else None
    try:
        runner = AlveoRunner(
            bitstream_path=XCLBIN_PATH,
            parameters=params,
            device=device,
            device_bus=DEVICE_BUS,
        )
        print("Runner initialized successfully.")
    except Exception as e:
        runner = None
        print(f"Failed to initialize AlveoRunner: {e}")

    XCLBIN_PATH, DEVICE_BUS, DEVICE_INDEX, runner
    return (runner,)


@app.cell
def _(ModelConfig, PATHS, TransformerAD, cuda_device, mo, os, torch):
    mo.md("### 5) Create `adv_trained_model.pt` on CPU/GPU")

    # 1. Define Paths and Configuration

    model_path = os.path.join(PATHS.experiments_dir, "adv_trained_model.pt")

    # This config matches the architecture of the model
    adv_model_config = ModelConfig(
        input_size=8,
        seq_len=12,
        embed_dim=32,      # Corresponds to old 'hidden_dim'
        latent_dim=16,     # Corresponds to old 'latent_dim' & 'dim_feedforward'
        num_heads=1,
        num_layers=1,
        dropout=0.1
    )

    # 2. Instantiate the new model architecture
    adv_model = TransformerAD(model_config=adv_model_config)
    print(f"Instantiated new TransformerAD model structure.")

    # 3. Load the state dictionary from the old model file
    print(f"Loading weights from: {model_path}")
    state_dict = torch.load(model_path, map_location=cuda_device)
    adv_model.load_state_dict(state_dict)
    adv_model.to(cuda_device)
    adv_model.eval()
    print("Successfully loaded 'adv_trained_model.pt' weights.")
    return adv_model, adv_model_config, state_dict


@app.cell
def _(
    FxPTransformerADConfig,
    FxpTransformerAD,
    adv_model_config,
    create_dynamic_qconfig,
    cuda_device,
    mo,
    state_dict,
    torch,
    val_dl,
):
    mo.md("### 6) Create FxPTransformerAD Model (W16A16)")

    # 1. Define QConfig for 32-bit weights and activations
    w16a16_qconfig = create_dynamic_qconfig(weight_bits=16, activation_bits=16)

    # 2. Instantiate FxP model
    # Use the same base config as the floating point model
    fxp_model_config = FxPTransformerADConfig(**adv_model_config.model_dump())
    fxp_model = FxpTransformerAD(config=fxp_model_config, q_config=w16a16_qconfig)

    # 3. Load weights from the pre-trained float model
    fxp_model.load_state_dict(state_dict)
    fxp_model.to(cuda_device)
    fxp_model.eval()

    print("Successfully instantiated FxpTransformerAD (W16A16) and loaded weights.")

    # 3. Calibrate the FxP Model"

    # a. Get one batch from validation dataloader for calibration
    calibration_batch = next(iter(val_dl))[0]["encoder_cont"]
    calibration_input = calibration_batch.to(cuda_device)
    print(f"Using one batch of validation data for calibration. Shape: {calibration_input.shape}")

    # b. Set weight quantization based on their values (no-overflow)
    fxp_model.set_min_mse_quant()
    print("Set min_mse fractional bits for weights.")

    # c. Run calibration pass to set activation quantization
    with torch.no_grad():
        _ = fxp_model(calibration_input, calibrate=True, calibration_type="min_mse")
    print("Ran calibration pass for activations.")

    # d. Permanently quantize weights and biases
    # fxp_model.quantize_weights_bias()
    # print("Permanently quantized model weights and biases.")
    return calibration_input, fxp_model


@app.cell
def _(
    THRESHOLD,
    adv_model,
    adv_model_config,
    approximation_comparison,
    cuda_device,
    fxp_model,
    mo,
    runner,
    test_dl,
):
    mo.md("### 8) Golden vs Approximated: Comparison Runs")

    import pandas as pd
    from privateer_ad.robustness.evaluator import evaluate_robustness, evaluate_robustness_alveo
    from privateer_ad.robustness.adversarial_dataloader import create_adversarial_dataloader

    loss_fn = "L1Loss"

    # Robustness settings
    ROBUSTNESS_EPS = 0.01
    ROBUSTNESS_STEPS = 100 # Iterations for PGD attack

    # --- 8.1: Run Standard Comparison (Float vs Alveo) ---
    print("1. Running Standard Evaluation (Clean Data)...")
    result_float_vs_alveo = approximation_comparison(
        golden_model=adv_model,
        approximated=runner,
        dataloader=test_dl,
        threshold=THRESHOLD,
        device=cuda_device,
        loss_fn_name=loss_fn,
    )

    # --- 8.2: Evaluate Robustness for Golden (Float) Model ---
    print("\n2. Evaluating Golden Model Robustness...")
    robustness_golden = evaluate_robustness(
        model=adv_model,
        model_config=adv_model_config,
        dataloader=test_dl,
        threshold=THRESHOLD,
        epsilons=[ROBUSTNESS_EPS],
        eps_step=0.0005,
        max_iter=ROBUSTNESS_STEPS,
        device=cuda_device
    )
    # Prefix metrics
    for k, v in robustness_golden.items():
        result_float_vs_alveo['golden']['metrics'][f"golden_{k}"] = v

    # --- 8.3: Evaluate Robustness for Alveo Model ---
    print("\n3. Evaluating Alveo Model Robustness...")
    robustness_alveo = evaluate_robustness_alveo(
        runner=runner,
        proxy_model=fxp_model, # Use FxP model for gradients
        model_config=adv_model_config,
        dataloader=test_dl,
        threshold=THRESHOLD,
        epsilons=[ROBUSTNESS_EPS],
        eps_step=0.0005,
        max_iter=ROBUSTNESS_STEPS,
        device=cuda_device
    )

    # Prefix metrics
    for k, v in robustness_alveo.items():
        # Map generic key to specific approx key
        result_float_vs_alveo['approx']['metrics'][f"approx_{k}"] = v

    # --- 8.4: Pretty Print Summary Table ---
    print("\n" + "="*80)
    print(f"{'COMPARISON SUMMARY':^80}")
    print("="*80)

    g_metrics = result_float_vs_alveo['golden']['metrics']
    a_metrics = result_float_vs_alveo['approx']['metrics']

    metric_map = {
        'ROC AUC': 'roc_auc',
        'Loss (L1)': 'loss',
        'Precision': 'precision',
        'Recall': 'recall',
        'F1-Score': 'f1-score',
        f'Robustness (ASR @ {ROBUSTNESS_EPS})': f'attack_success_rate_eps_{ROBUSTNESS_EPS}'
    }

    table_data = []
    for display_name, base_key in metric_map.items():
        g_key = f"golden_{base_key}"
        a_key = f"approx_{base_key}"
    
        g_val = g_metrics.get(g_key, 0.0)
        a_val = a_metrics.get(a_key, 0.0)
        diff = a_val - g_val
    
        table_data.append({
            "Metric": display_name,
            "Golden (Float)": g_val,
            "Approx (Alveo)": a_val,
            "Diff": diff
        })

    df_res = pd.DataFrame(table_data)

    format_mapping = {
        "Golden (Float)": "{:,.4f}", 
        "Approx (Alveo)": "{:,.4f}", 
        "Diff": "{:+,.4f}"
    }

    print(df_res.to_string(index=False, formatters={
        k: v.format for k, v in format_mapping.items()
    }))
    print("-" * 80)

    print("\nPer-Sample Difference Summary:")
    print(pd.Series(result_float_vs_alveo['comparison']['summary']).to_string())
    return (result_float_vs_alveo,)


@app.cell
def _(result_float_vs_alveo):
    print(result_float_vs_alveo['golden'])
    print(result_float_vs_alveo['approx'])
    print(result_float_vs_alveo['comparison']['summary'])
    return


@app.cell
def _(mo, result_float_vs_alveo, result_float_vs_fxp, result_fxp_vs_alveo):
    mo.md("### 9) Comparison Figures")
    result_float_vs_fxp['comparison']['figures'],  result_float_vs_alveo['comparison']['figures'], result_fxp_vs_alveo['comparison']['figures'], 
    return


@app.cell
def _(
    PATHS,
    os,
    result_float_vs_alveo,
    result_float_vs_fxp,
    result_fxp_vs_alveo,
):
    import plotly.graph_objects as go
    import matplotlib.pyplot as plt

    # 1. Define the output directory
    figures_dir = os.path.join(PATHS.experiments_dir, "w16a16_figures")
    os.makedirs(figures_dir, exist_ok=True)

    print(f"Saving figures to: {figures_dir}")

    # 2. Helper function to save figuresf
    def save_scenario_figures(result_dict, file_prefix):
        figs = result_dict["comparison"]["figures"]

        # If 'figs' is a dictionary (e.g. {'name': fig}), iterate items.
        # If it's a list, iterate with index.
        if isinstance(figs, dict):
            iterator = figs.items() # (name, fig_obj)
        elif isinstance(figs, (list, tuple)):
            iterator = enumerate(figs) # (index, fig_obj)
        else:
            iterator = [(0, figs)] # Single object

        for identifier, fig in iterator:
            # Clean filename
            clean_id = str(identifier).replace(" ", "_").replace("/", "-")
            filename = f"{file_prefix}_{clean_id}.png"
            save_path = os.path.join(figures_dir, filename)

            try:
                # CASE A: It is a Dictionary (Plotly dict or wrapper)67:8080
                if isinstance(fig, dict):
                    # Check if it's a Plotly serialization
                    if 'data' in fig and 'layout' in fig:
                        fig_obj = go.Figure(fig)
                        fig_obj.write_image(save_path)
                        print(f"Saved (Plotly Dict): {filename}")
                    else:
                        print(f"Skipped {filename}: Dict structure unknown (keys: {list(fig.keys())})")

                # CASE B: It is a Matplotlib Figure
                elif hasattr(fig, 'savefig'):
                    fig.savefig(save_path, bbox_inches='tight', dpi=300)
                    print(f"Saved (Matplotlib): {filename}")

                # CASE C: It is a Plotly Figure object
                elif hasattr(fig, 'write_image'):
                    fig.write_image(save_path)
                    print(f"Saved (Plotly Object): {filename}")

                else:
                    print(f"Skipped {filename}: Unknown type {type(fig)}")

            except Exception as e:
                print(f"Error saving {filename}: {e}")

    # 3. Save each scenario
    save_scenario_figures(result_float_vs_fxp, "float_vs_fxp")
    save_scenario_figures(result_float_vs_alveo, "float_vs_alveo")
    save_scenario_figures(result_fxp_vs_alveo, "fxp_vs_alveo")
    return


@app.cell
def _(runner):
    runner.clean_class()
    return


@app.cell
def _(calibration_input, fxp_model, mo, os):
    mo.md("### 10) Save Approximation Weights and QConfig for use in HLS")

    from privateer_ad.approximation.utils.export_weights import (
            convert_model_to_json_ae,
            convert_json_to_h_ae,
            post_process_header_for_specific_types,
        )

    OUTPUT_DIR = os.path.join("app", "outputs", "fxp_w16a16_no_w_quant")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save the final, calibrated qconfig for inspection
    qconfig_w16_a16_path = os.path.join(
        OUTPUT_DIR, "adv_model_w16_a16_qconfig.json"
    )
    with open(qconfig_w16_a16_path, "w") as f:
        f.write(fxp_model.q_config.model_dump_json(indent=2))
    print(f"\nSaved calibrated QConfig to {qconfig_w16_a16_path}")

    # Export weights to JSON and H files
    json_path_w16_a16 = os.path.join(OUTPUT_DIR , "adv_model_w16_a16.json")
    h_path_w16_a16 = os.path.join(OUTPUT_DIR , "adv_model_w16_a16.h")

    print(
        f"\nExporting float model weights to {json_path_w16_a16} and {h_path_w16_a16}..."
    )
    convert_model_to_json_ae(
        fxp_model,
        filename=json_path_w16_a16,
        input_shape=calibration_input.shape,
    )
    convert_json_to_h_ae(json_filename=json_path_w16_a16, h_filename=h_path_w16_a16)
    post_process_header_for_specific_types(h_filename=h_path_w16_a16)
    return


if __name__ == "__main__":
    app.run()
