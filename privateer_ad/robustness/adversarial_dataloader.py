# privateer_ad/robustness/adversarial_dataloader.py
import logging
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from privateer_ad.config import ModelConfig
from privateer_ad.robustness.attacker import create_adversarial_attacker

# Suppress verbose logging from the ART library
logging.getLogger("art").setLevel(logging.WARNING)


class AdversarialDataset(Dataset):
    """
    A PyTorch Dataset to wrap adversarial inputs and original labels,
    formatting them to be compatible with the existing Dataloaders.
    """

    def __init__(self, adv_inputs: torch.Tensor, original_labels: torch.Tensor):
        self.adv_inputs = adv_inputs
        self.original_labels = original_labels

    def __len__(self) -> int:
        return len(self.adv_inputs)

    def __getitem__(self, idx) -> Tuple[Dict[str, torch.Tensor], Tuple[torch.Tensor, ...]]:
        """
        Returns data in a nested tuple format to match the output of
        pytorch-forecasting's TimeSeriesDataSet.to_dataloader().

        This ensures 100% compatibility with the existing ModelTrainer, which
        expects to access data via `batch[0]['encoder_cont']` and `batch[1][0]`.
        """
        inputs_dict = {"encoder_cont": self.adv_inputs[idx]}
        # The labels need to be wrapped in a tuple as well.
        labels_tuple = (self.original_labels[idx],)
        return inputs_dict, labels_tuple


def create_adversarial_dataloader(
    dataloader: DataLoader,
    model: torch.nn.Module,
    model_config: ModelConfig,
    threshold: float,
    eps: float,
    eps_step: float,
    max_iter: int,
    device: str,
) -> DataLoader:
    """
    Generates a new DataLoader containing adversarial examples.

    This function iterates through an existing dataloader, applies a PGD attack
    to each batch of inputs, and then collects the adversarial examples into a
    new dataloader compatible with the project's ModelTrainer.

    Args:
        dataloader: The original DataLoader with benign and malicious samples.
        model: The model to be attacked.
        model_config: The configuration of the model.
        threshold: The anomaly detection threshold for the model.
        eps: The maximum perturbation for the PGD attack.
        eps_step: The step size for each iteration of the PGD attack.
        max_iter: The number of iterations for the PGD attack.
        device: The device to run the attack on ('cuda' or 'cpu').

    Returns:
        A new DataLoader containing the generated adversarial examples and their
        original corresponding labels, formatted for the ModelTrainer.
    """
    logging.info(f"Creating adversarial dataloader with eps={eps}, max_iter={max_iter}...")

    attacker = create_adversarial_attacker(
        model, model_config, threshold, eps, eps_step, max_iter, device, dataloader.batch_size
    )

    adversarial_samples = []
    original_labels_list = []

    # Iterate through the original dataloader to generate attacks
    for batch in tqdm(dataloader, desc=f"Generating adversarial examples (eps={eps})"):
        # The structure of the batch from pytorch-forecasting is handled here
        inputs = batch[0]["encoder_cont"].to(device)
        labels = batch[1][0].squeeze().long().to(device)

        # The attack target is the opposite of the original label.
        # Attack benign (0) to become malicious (1), and malicious (1) to become benign (0).
        targets = 1 - labels

        # ART library works with numpy arrays
        adversarial_inputs_np = attacker.generate(
            x=inputs.cpu().numpy(), y=targets.cpu().numpy()
        )

        adversarial_samples.append(torch.from_numpy(adversarial_inputs_np))
        original_labels_list.append(labels.cpu())

    # Combine all generated samples and labels
    all_adversarial_samples = torch.cat(adversarial_samples, dim=0)
    all_original_labels = torch.cat(original_labels_list, dim=0)

    # Create a new custom Dataset and DataLoader
    adv_dataset = AdversarialDataset(all_adversarial_samples, all_original_labels)

    # Preserve original dataloader settings like batch_size, num_workers etc.
    adv_dataloader = DataLoader(
        adv_dataset,
        batch_size=dataloader.batch_size,
        shuffle=True,  # Shuffle the adversarial dataset for training
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
    )

    logging.info("Adversarial dataloader created successfully.")
    return adv_dataloader