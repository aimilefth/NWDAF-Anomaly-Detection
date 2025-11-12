# privateer_ad/dse/ptq_dse.py
import os
import json
import time
from datetime import datetime
import itertools
import logging

import mlflow
import torch

from privateer_ad.config import (
    DataConfig,
    ModelConfig,
    MLFlowConfig,
    PathConfig,
)
from privateer_ad.old_data_utils import OldDataProcessor
from privateer_ad.architectures import TransformerAD
from privateer_ad.approximation.transformer_ad_fxp import (
    FxpTransformerAD,
    FxPTransformerADConfig,
    create_dynamic_qconfig,
)
from privateer_ad.approximation.approximation_comparison import approximation_comparison_mlflow
from privateer_ad.evaluate import ModelEvaluator
from privateer_ad.robustness.evaluator import evaluate_robustness


logging.basicConfig(level=logging.INFO)


def ptq_dse():
    """
    Performs a Design Space Exploration (DSE) for Post-Training Quantization (PTQ)
    on the TransformerAD model.
    """
    # --- Fixed Configurations for the DSE ---
    PATHS = PathConfig()
    MLFLOW_CONFIG = MLFlowConfig()
    DATA_CONFIG = DataConfig(seq_len=12)
    
    # This config MUST match the architecture of the saved `adv_trained_model.pt`
    # Based on your .env and marimo notebook, these are the correct values.
    PRETRAINED_MODEL_CONFIG = ModelConfig(
        model_name="TransformerAD_Golden", # Give it a distinct name for clarity
        input_size=8,
        seq_len=DATA_CONFIG.seq_len,
        embed_dim=32,
        latent_dim=16,
        num_heads=1,
        num_layers=1,
        dropout=0.2,
    )
    
    # Path to the saved floating-point model checkpoint
    SAVED_FLOAT_MODEL_PATH = PATHS.experiments_dir / "adv_trained_model.pt"
    if not SAVED_FLOAT_MODEL_PATH.exists():
        logging.error(f"Pre-trained model not found at: {SAVED_FLOAT_MODEL_PATH}")
        return

    # Use a fixed threshold for all runs for fair comparison.
    # This value is from your marimo notebook.
    THRESHOLD = 0.0209596287459135

    # --- DSE Grid Parameters ---
    weight_bits_options = [16, 14, 12, 10]
    activation_bits_options = [16, 14, 12, 10]
    weight_quant_type_options = ["min_mse"]
    calibration_type_options = ["min_mse"]
    test_batch_size = 2048
    calibration_batch_size = 131072

    # --- Robustness Parameters ---
    EPSILONS = [0.01]
    EPS_STEP = 0.0005
    MAX_ITER = 100

    # --- DSE Setup ---
    mlflow.set_tracking_uri(MLFLOW_CONFIG.tracking_uri)
    mlflow.set_experiment(f"PTQ_DSE_ADV_{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Data ---
    logging.info("Setting up data loaders...")
    dp = OldDataProcessor()
    val_dl = dp.get_dataloader("val", batch_size=calibration_batch_size, only_benign=True, num_workers=16)
    test_dl = dp.get_dataloader("test", batch_size=test_batch_size, only_benign=False, num_workers=16)
    calibration_input = next(iter(val_dl))[0]["encoder_cont"].to(device)
    calibration_real_batch_size = calibration_input.shape[0]

    # --- Load Golden Model ---
    logging.info(f"Loading golden model from {SAVED_FLOAT_MODEL_PATH}")
    golden_model = TransformerAD(model_config=PRETRAINED_MODEL_CONFIG)
    state_dict = torch.load(SAVED_FLOAT_MODEL_PATH, map_location=device)
    golden_model.load_state_dict(state_dict)
    golden_model.to(device)
    golden_model.eval()
    logging.info("Golden model loaded successfully.")

    # --- Pre-calculate Golden Model results to avoid re-computation ---
    logging.info("Pre-calculating evaluation results for the golden model...")
    golden_evaluator = ModelEvaluator(device=device, loss_fn="L1Loss")
    golden_metrics, golden_figures, golden_scores = golden_evaluator.evaluate(
        model=golden_model,
        dataloader=test_dl,
        threshold=THRESHOLD,
        prefix="golden",
        return_anomaly_scores=True
    )
    calculated_golden = (golden_metrics, golden_figures, golden_scores)
    
    logging.info("Pre-calculating robustness for the golden model...")
    calculated_golden_robustness = evaluate_robustness(
        model=golden_model,
        model_config=PRETRAINED_MODEL_CONFIG,
        dataloader=test_dl,
        threshold=THRESHOLD,
        epsilons=EPSILONS,
        eps_step=EPS_STEP,
        max_iter=MAX_ITER,
        device=device,
    )
    logging.info(f"Golden model robustness pre-calculated: {calculated_golden_robustness}")

    # --- DSE Loop ---
    dse_configurations = list(itertools.product(
        weight_bits_options,
        activation_bits_options,
        weight_quant_type_options,
        calibration_type_options,
    ))
    total_runs = len(dse_configurations)
    results_summary = []

    logging.info(f"Starting PTQ DSE with {total_runs} configurations.")

    for idx, (wb, ab, w_q_type, cal_type) in enumerate(dse_configurations, start=1):
        
        dse_params = {
            "weight_bits": wb,
            "activation_bits": ab,
            "weight_quant_type": w_q_type,
            "calibration_type": cal_type,
            "calibration_batch_size": calibration_real_batch_size,
        }
        
        run_name = (
            f"PTQ_W{wb}_A{ab}_{w_q_type}_{cal_type}_CalBS{calibration_real_batch_size}_"
            f"{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        
        logging.info(f"--- Starting DSE run {idx}/{total_runs}: {run_name} ---")
        start_time = time.perf_counter()

        # 1. Create Quantized Model for this run
        q_config = create_dynamic_qconfig(weight_bits=wb, activation_bits=ab)
        fxp_model_config = FxPTransformerADConfig(**PRETRAINED_MODEL_CONFIG.model_dump())
        approximated_model = FxpTransformerAD(config=fxp_model_config, q_config=q_config)
        approximated_model.load_state_dict(state_dict)
        approximated_model.to(device)
        approximated_model.eval()

        # 2. Apply PTQ
        logging.info("Applying post-training quantization...")
        if w_q_type == "no_overflow":
            approximated_model.set_no_overflow_quant()
        elif w_q_type == "min_mse":
            approximated_model.set_min_mse_quant()

        # 3. Calibrate
        logging.info(f"Calibrating with {calibration_real_batch_size} samples...")
        with torch.no_grad():
            _ = approximated_model(calibration_input, calibrate=True, calibration_type=cal_type)
        logging.info("Calibration finished.")
        
        # 4. Evaluate and Log
        result = approximation_comparison_mlflow(
            golden_model=golden_model,
            approximated=approximated_model,
            dataloader=test_dl,
            threshold=THRESHOLD,
            mlflow_run_name=run_name,
            mlflow_params=dse_params,
            q_config=approximated_model.q_config,
            device=device,
            check_robustness_approx=True,
            check_robustness_golden=True,
            epsilons=EPSILONS,
            calculated_golden=calculated_golden,
            calculated_golden_robustness=calculated_golden_robustness,
        )

        end_time = time.perf_counter()
        elapsed = end_time - start_time
        logging.info(f"Finished DSE run {idx} in {elapsed:.2f} seconds.")

        results_summary.append({
            "run_name": run_name,
            "dse_params": dse_params,
            "metrics": result.get("approx", {}).get("metrics", {}),
            "comparison_summary": result.get("comparison", {}).get("summary", {}),
            "elapsed_time_sec": elapsed,
            "error": result.get("error")
        })

    # Save summary to a local file
    results_filename = f"ptq_dse_results_{datetime.now().strftime('%Y%m%d')}.json"
    results_path = os.path.join(PATHS.experiments_dir, results_filename)
    with open(results_path, "w") as f:
        json.dump(results_summary, f, indent=4)
    logging.info(f"PTQ DSE complete. Summary saved to {results_path}")


if __name__ == "__main__":
    ptq_dse()