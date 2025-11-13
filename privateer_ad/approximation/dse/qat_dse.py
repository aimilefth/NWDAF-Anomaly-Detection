# privateer_ad/approximation/dse/qat_dse.py
import os
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
    TrainingConfig,
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
from privateer_ad.train.trainer import ModelTrainer
from privateer_ad.robustness.adversarial_dataloader import create_adversarial_dataloader

logging.basicConfig(level=logging.INFO)


def qat_dse(adversarial_training: bool = False):
    """
    Performs a Design Space Exploration (DSE) for Quantization-Aware Training (QAT)
    on the TransformerAD model, with an option for adversarial training.
    """
    # --- Fixed Configurations for the DSE ---
    PATHS = PathConfig()
    MLFLOW_CONFIG = MLFlowConfig()
    DATA_CONFIG = DataConfig(seq_len=12)
    
    PRETRAINED_MODEL_CONFIG = ModelConfig(
        model_name="TransformerAD_Golden",
        input_size=8,
        seq_len=DATA_CONFIG.seq_len,
        embed_dim=32,
        latent_dim=16,
        num_heads=1,
        num_layers=1,
        dropout=0.2,
    )
    
    SAVED_FLOAT_MODEL_PATH = PATHS.experiments_dir / "adv_trained_model.pt"
    if not SAVED_FLOAT_MODEL_PATH.exists():
        logging.error(f"Pre-trained model not found at: {SAVED_FLOAT_MODEL_PATH}")
        return

    THRESHOLD = 0.0209596287459135

    # --- DSE Grid Parameters ---
    weight_bits_options = [16, 14, 12, 10]
    activation_bits_options = [16, 14, 12, 10]
    weight_quant_type_options = ["min_mse"]
    calibration_type_options = ["min_mse"]
    test_batch_size = 2048
    calibration_batch_size = 131072
    qat_epochs = 100
    
    # --- Robustness Training Parameters ---
    TR_EPSILON = 0.01
    TR_EPS_STEP = 0.0005
    TR_MAX_ITER = 100

    # --- Robustness Evaluation Parameters ---
    EV_EPSILONS = [0.01]
    EV_EPS_STEP = 0.0005
    EV_MAX_ITER = 100
    # --- DSE Setup ---
    adv_suffix = "QAT_ADV" if adversarial_training else "QAT"
    mlflow.set_tracking_uri(MLFLOW_CONFIG.tracking_uri)
    mlflow.set_experiment(f"{adv_suffix}_DSE_{datetime.now().strftime('%Y%m%d-%H%M%S')}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Load Data ---
    logging.info("Setting up data loaders...")
    dp = OldDataProcessor()
    train_dl = dp.get_dataloader("train", batch_size=4096, only_benign=True, num_workers=16)
    val_dl = dp.get_dataloader("val", batch_size=4096, only_benign=True, num_workers=16)
    test_dl = dp.get_dataloader("test", batch_size=4096, only_benign=False, num_workers=16)
    calibration_dl = dp.get_dataloader("val", batch_size=131072, only_benign=True, num_workers=16)
    calibration_input = next(iter(calibration_dl))[0]["encoder_cont"].to(device)
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
        epsilons=EV_EPSILONS,
        eps_step=EV_EPS_STEP,
        max_iter=EV_MAX_ITER,
        device=device,
    )
    logging.info(f"Golden model robustness pre-calculated: {calculated_golden_robustness}")

    # (Optional) Create Adversarial DataLoaders for training
    train_dl_to_use, val_dl_to_use = train_dl, val_dl
    if adversarial_training:
        logging.info("Adversarial training enabled. Generating adversarial dataloaders...")
        adv_train_dl = create_adversarial_dataloader(
            dataloader=train_dl, model=golden_model, model_config=PRETRAINED_MODEL_CONFIG,
            threshold=THRESHOLD, eps=TR_EPSILON, eps_step=TR_EPS_STEP, max_iter=TR_MAX_ITER, device=device
        )
        adv_val_dl = create_adversarial_dataloader(
            dataloader=val_dl, model=golden_model, model_config=PRETRAINED_MODEL_CONFIG,
            threshold=THRESHOLD, eps=TR_EPSILON, eps_step=TR_EPS_STEP, max_iter=TR_MAX_ITER, device=device
        )
        train_dl_to_use, val_dl_to_use = adv_train_dl, adv_val_dl


    # --- DSE Loop ---
    dse_configurations = list(itertools.product(
        weight_bits_options,
        activation_bits_options,
        weight_quant_type_options,
        calibration_type_options,
    ))
    total_runs = len(dse_configurations)
    results_summary = []

    logging.info(f"Starting {adv_suffix} DSE with {total_runs} configurations.")

    for idx, (wb, ab, w_q_type, cal_type) in enumerate(dse_configurations, start=1):        
        
        dse_params = {
            "weight_bits": wb,
            "activation_bits": ab,
            "weight_quant_type": w_q_type,
            "calibration_type": cal_type,
            "calibration_batch_size": calibration_real_batch_size,
            "qat_epochs": qat_epochs,
            "adversarial_training": adversarial_training,
        }
        if adversarial_training:
            dse_params.update({"training_adv_eps": TR_EPSILON, "training_adv_eps_step": TR_EPS_STEP ,"training_adv_max_iter": TR_MAX_ITER})

        run_name = f"{adv_suffix}_W{wb}_A{ab}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        logging.info(f"--- Starting DSE run {idx}/{total_runs}: {run_name} ---")
        
        # Use a 'with' statement to manage the MLflow run for this trial
        with mlflow.start_run(run_name=run_name) as run:
            start_time = time.perf_counter()
            mlflow.log_params(dse_params) # Log DSE params at the start of the run

            # 1. Create and initialize Quantized Model
            q_config = create_dynamic_qconfig(weight_bits=wb, activation_bits=ab)
            fxp_model_config = FxPTransformerADConfig(**PRETRAINED_MODEL_CONFIG.model_dump())
            approximated_model = FxpTransformerAD(config=fxp_model_config, q_config=q_config)
            approximated_model.load_state_dict(state_dict)
            approximated_model.to(device)

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
            
            # 4. Perform Quantization-Aware Training (Fine-tuning)
            logging.info(f"Starting QAT for {qat_epochs} epochs...")
            qat_training_config = TrainingConfig(epochs=qat_epochs, target_metric='val_loss', direction='minimize')
            optimizer = torch.optim.Adam(approximated_model.parameters(), lr=qat_training_config.learning_rate)
            
            trainer = ModelTrainer(
                model=approximated_model,
                optimizer=optimizer,
                device=device,
                training_config=qat_training_config
            )
            
            # The forward pass of FxpTransformerAD with apply_ste=True handles QAT
            # The trainer will call this model's forward pass automatically
            best_checkpoint = trainer.training(train_dl=train_dl_to_use, val_dl=val_dl_to_use)
            
            # Load the best state found during fine-tuning
            approximated_model.load_state_dict(best_checkpoint['model_state_dict'])
            approximated_model.eval()
            logging.info("QAT finished.")

            # 4. Evaluate and Log to the *same* active MLflow run
            _ = approximation_comparison_mlflow(
                golden_model=golden_model,
                approximated=approximated_model,
                dataloader=test_dl,
                threshold=THRESHOLD,
                mlflow_run_name=run_name, # Name is for legacy, not used here
                mlflow_params=dse_params, # Params are logged again, but MLflow handles it
                q_config=approximated_model.q_config,
                device=device,
                check_robustness_approx=True,
                check_robustness_golden=True,
                epsilons=EV_EPSILONS,
                eps_step=EV_EPS_STEP,
                max_iter=EV_MAX_ITER,
                calculated_golden=calculated_golden,
                calculated_golden_robustness=calculated_golden_robustness,
                manage_run=False # <--- THIS IS THE KEY CHANGE
            )

            end_time = time.perf_counter()
            logging.info(f"Finished DSE run {idx} in {end_time - start_time:.2f} seconds.")

    logging.info(f"{adv_suffix} DSE complete.")


if __name__ == "__main__":
    qat_dse(adversarial_training=True)