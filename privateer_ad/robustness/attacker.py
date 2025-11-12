from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
from art.attacks.evasion import ProjectedGradientDescentPyTorch
from art.estimators.classification import PyTorchClassifier

from privateer_ad.config import ModelConfig


class ModelWrapperForART(nn.Module):
    """
    A wrapper for the TransformerAD model to make it compatible with the ART library for classification tasks.
    This wrapper takes the reconstruction output of the TransformerAD model and computes an anomaly score,
    which is then converted into a class probability.
    """

    def __init__(
        self,
        model: nn.Module,
        threshold: float,
    ):
        super().__init__()
        self.model = model
        self.model.eval()
        self.threshold = threshold
        self.criterion = nn.L1Loss(reduction="none")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reconstructed_x = self.model(x)
        reconstruction_error = self.criterion(reconstructed_x, x).mean(dim=(1, 2))
        # Convert the reconstruction error to a "malicious" probability.
        # A higher error means a higher probability of being malicious.
        # We use a sigmoid to keep the output in the [0, 1] range.
        anomaly_prob = torch.sigmoid(
            (reconstruction_error - self.threshold) / self.threshold
        )
        return anomaly_prob


def create_adversarial_attacker(
    model: nn.Module,
    model_config: ModelConfig,
    threshold: float,
    eps: float = 0.01,
    eps_step: float = 0.0005,
    max_iter: int = 100,
    device: str = 'cpu',
    batch_size: int = 4096,
) -> ProjectedGradientDescentPyTorch:
    """
    Creates and configures a Projected Gradient Descent (PGD) attacker from the ART library.

    Args:
        model: The PyTorch model to be attacked.
        model_config: The configuration of the model.
        threshold: The anomaly detection threshold.
        eps: The maximum perturbation for the attack.
        max_iter: The number of iterations for the attack.
        device: The device to run the attack on ('cuda' or 'cpu').

    Returns:
        An instance of ProjectedGradientDescentPyTorch.
    """
    art_model_wrapper = ModelWrapperForART(model, threshold)
    art_classifier = PyTorchClassifier(
        model=art_model_wrapper,
        loss=nn.BCELoss(),
        input_shape=(model_config.seq_len, model_config.input_size),
        nb_classes=2,
        device_type=device,
    )

    return ProjectedGradientDescentPyTorch(
        art_classifier,
        eps=eps,
        eps_step=eps_step,
        norm="inf",
        max_iter=max_iter,
        targeted=True,
        batch_size=batch_size,
        verbose=False,
    )