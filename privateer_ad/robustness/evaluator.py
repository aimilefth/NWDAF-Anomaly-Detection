import logging
from typing import Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from privateer_ad.config import ModelConfig
from privateer_ad.robustness.attacker import create_adversarial_attacker
from privateer_ad.alveo.alveo_runner import AlveoRunner

# Suppress verbose logging from the ART library
logging.getLogger("art").setLevel(logging.WARNING)

def evaluate_robustness(
    model: Union[nn.Module, torch.nn.Module],
    model_config: ModelConfig,
    dataloader: DataLoader,
    threshold: float,
    epsilons: List[float],
    eps_step: float,
    max_iter: int,
    device: str,
) -> Dict[str, float]:
    """
    Evaluates the adversarial robustness of a model for different epsilon values.

    Args:
        model: The model to evaluate.
        model_config: The configuration of the model.
        dataloader: The dataloader with the test data.
        threshold: The anomaly detection threshold.
        epsilons: A list of epsilon values to test.
        device: The device to run the evaluation on.

    Returns:
        A dictionary with robustness metrics for each epsilon.
    """
    robustness_metrics = {}
    batch_size = dataloader.batch_size
    for eps in epsilons:
        successful_attacks = 0
        total_samples = 0

        # Create a new attacker for each epsilon value
        attacker = create_adversarial_attacker(
            model, model_config, threshold, eps, eps_step=eps_step, max_iter=max_iter, device=device, batch_size=batch_size
        )

        for batch in tqdm(dataloader, desc=f"Attacking with eps={eps}"):
            # Correctly unpack the batch from the pytorch-forecasting dataloader
            # The input tensor is in a dictionary, and labels are in a tuple.
            inputs = batch[0]["encoder_cont"]
            labels = batch[1][0].squeeze().long()

            inputs, labels = inputs.to(device), labels.to(device)
            # Generate adversarial examples. We attack benign samples to make them malicious (target=1)
            # and malicious samples to make them benign (target=0).
            targets = 1 - labels

            adversarial_inputs = torch.from_numpy(
                attacker.generate(x=inputs.cpu().numpy(), y=targets.cpu().numpy())
            ).to(device)

            # Evaluate the model on the original and adversarial inputs
            with torch.no_grad():
                original_outputs = model(inputs)
                adversarial_outputs = model(adversarial_inputs)

            original_errors = (
                torch.abs(original_outputs - inputs).mean(dim=(1, 2)).cpu().numpy()
            )
            adversarial_errors = (
                torch.abs(adversarial_outputs - adversarial_inputs).mean(dim=(1, 2)).cpu().numpy()
            )

            original_predictions = (original_errors > threshold).astype(int)
            adversarial_predictions = (adversarial_errors > threshold).astype(int)

            # An attack is successful if the prediction changes
            successful_attacks += np.sum(original_predictions != adversarial_predictions)
            total_samples += len(inputs)

        attack_success_rate = (
            successful_attacks / total_samples if total_samples > 0 else 0
        )
        robustness_metrics[f"attack_success_rate_eps_{eps}"] = attack_success_rate

    return robustness_metrics


def evaluate_robustness_alveo(
    runner: AlveoRunner,
    proxy_model: Union[nn.Module, torch.nn.Module],
    model_config,
    dataloader: DataLoader,
    threshold: float,
    epsilons: List[float],
    eps_step: float,
    max_iter: int,
    device: str,
) -> Dict[str, float]:
    """
    Evaluates adversarial robustness using an Alveo FPGA Runner for inference,
    while using a proxy PyTorch model to generate gradients for the attack.
    """
    robustness_metrics = {}
    batch_size = dataloader.batch_size
    for eps in epsilons:
        successful_attacks = 0
        total_samples = 0

        # 1. Create Attacker using the PROXY model (FxP or Float)
        # This allows ART to compute gradients which are impossible on the FPGA
        attacker = create_adversarial_attacker(
            proxy_model, model_config, threshold, eps, eps_step=eps_step, max_iter=max_iter, device=device, batch_size=batch_size
        )

        for batch in tqdm(dataloader, desc=f"Alveo Robustness (eps={eps})"):
            # Unpack batch
            inputs = batch[0]["encoder_cont"] # [B, Seq, Feat]
            labels = batch[1][0].squeeze().long()
            
            inputs = inputs.to(device)
            
            # 2. Generate Adversarial Examples (Targeted: Benign<->Malicious)
            targets = 1 - labels.to(device)
            adversarial_inputs_np = attacker.generate(x=inputs.cpu().numpy(), y=targets.cpu().numpy())
            
            # 3. Prepare data for Alveo Runner
            # Inputs need to be numpy for pynq/xrt
            inputs_np = inputs.cpu().numpy()
            B, T, F = inputs_np.shape
            
            # 4. Run Inference on Alveo (original vs Adversarial)
            # Note: run_vector handles flattening and reshaping internally if output_shape is passed
            
            # Clean Inference
            original_outputs_np = runner.run_vector(
                input_vector=inputs_np,
                output_shape=(B, T, F),
                timed=False,
                verbose=False
            )
            
            # Adversarial Inference
            adversarial_outputs_np = runner.run_vector(
                input_vector=adversarial_inputs_np,
                output_shape=(B, T, F),
                timed=False,
                verbose=False
            )

            # 5. Compute Reconstruction Errors (in PyTorch for convenience)
            inputs = torch.from_numpy(inputs_np).to(device)
            adversarial_inputs = torch.from_numpy(adversarial_inputs_np).to(device)
            original_outputs = torch.from_numpy(original_outputs_np).to(device)
            adversarial_outputs = torch.from_numpy(adversarial_outputs_np).to(device)

            original_errors = (
                torch.abs(original_outputs - inputs).mean(dim=(1, 2)).cpu().numpy()
            )
            adversarial_errors = (
                torch.abs(adversarial_outputs - adversarial_inputs).mean(dim=(1, 2)).cpu().numpy()
            )

            original_predictions = (original_errors > threshold).astype(int)
            adversarial_predictions = (adversarial_errors > threshold).astype(int)

            # An attack is successful if the prediction changes
            successful_attacks += np.sum(original_predictions != adversarial_predictions)
            total_samples += len(inputs)

        attack_success_rate = (
            successful_attacks / total_samples if total_samples > 0 else 0
        )
        robustness_metrics[f"attack_success_rate_eps_{eps}"] = attack_success_rate

    return robustness_metrics