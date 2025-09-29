import numpy as np
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from .transform import OldDataProcessor

class DeviceSequenceDataset(Dataset):
    def __init__(self, feature_sequences: List[np.ndarray], label_sequences: List[np.ndarray]):
        if not feature_sequences:
            raise ValueError("feature_sequences cannot be empty.")
        if len(feature_sequences) != len(label_sequences):
            raise ValueError("Mismatch between number of feature sequences and label sequences.")
        self.feature_sequences = [torch.tensor(seq, dtype=torch.float32) for seq in feature_sequences]
        self.label_sequences = [torch.tensor(labels, dtype=torch.long) for labels in label_sequences]

    def __len__(self) -> int:
        return len(self.feature_sequences)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {"encoder_cont": self.feature_sequences[idx],
                "attack_labels": self.label_sequences[idx]}

def _resolve_features(dp: OldDataProcessor, df, use_pca: bool):
    if use_pca:
        if dp.pca is None:
            try:
                dp.load_pca()
            except FileNotFoundError:
                raise FileNotFoundError(f"PCA model not found at {dp.pca_path}")
        return dp.pca.get_feature_names_out().tolist()
    return dp.input_features

def create_non_overlapping_device_dataloader(
    data_processor: OldDataProcessor,
    device_imeisv_target: str,
    dataset_split: str = "val",
    use_pca: bool = False,
    seq_len: int = 12,
    batch_size: int = 32,
    max_sequences_to_extract: Optional[int] = None,
    num_workers: int = 0,
    shuffle: bool = False,
    sequence_offset: int = 0,
) -> Optional[DataLoader]:

    df = data_processor.preprocess_data(path=dataset_split, use_pca=use_pca, setup=False)
    if "imeisv" not in df.columns or "attack" not in df.columns:
        print("Expected columns 'imeisv' and 'attack' not found.")
        return None
    df["imeisv"] = df["imeisv"].astype(str)

    device_df = df[df["imeisv"] == str(device_imeisv_target)].copy()
    if device_df.empty:
        print(f"No data for device {device_imeisv_target}.")
        return None

    feature_names = _resolve_features(data_processor, device_df, use_pca)
    features, labels, n_extracted = [], [], 0

    start = sequence_offset * seq_len
    for i in range(start, len(device_df), seq_len):
        end = i + seq_len
        if end > len(device_df):
            break
        slice_df = device_df.iloc[i:end]
        features.append(slice_df[feature_names].values)
        labels.append(slice_df["attack"].values.astype(int))
        n_extracted += 1
        if max_sequences_to_extract is not None and n_extracted >= max_sequences_to_extract:
            break

    if n_extracted == 0:
        print("No sequences extracted.")
        return None

    dataset = DeviceSequenceDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=bool(num_workers > 0 and torch.cuda.is_available()))

def create_overlapping_device_dataloader(
    data_processor: OldDataProcessor,
    device_imeisv_target: str,
    dataset_split: str = "val",
    use_pca: bool = False,
    seq_len: int = 12,
    batch_size: int = 32,
    max_sequences_to_extract: Optional[int] = None,
    num_workers: int = 0,
    shuffle: bool = False,
    sequence_offset: int = 0,  # offset in timesteps
    logger: callable = print,
) -> Optional[DataLoader]:

    df = data_processor.preprocess_data(path=dataset_split, use_pca=use_pca, setup=False)
    if "imeisv" not in df.columns or "attack" not in df.columns:
        logger("Expected columns 'imeisv' and 'attack' not found.")
        return None
    df["imeisv"] = df["imeisv"].astype(str)

    device_df = df[df["imeisv"] == str(device_imeisv_target)].copy()
    if device_df.empty:
        logger(f"No data for device {device_imeisv_target}.")
        return None

    if len(device_df) < seq_len + sequence_offset:
        logger("Not enough data for the requested offset + seq_len.")
        return None

    feature_names = _resolve_features(data_processor, device_df, use_pca)
    features, labels, n_extracted = [], [], 0

    for i in range(sequence_offset, len(device_df) - seq_len + 1):
        j = i + seq_len
        slice_df = device_df.iloc[i:j]
        features.append(slice_df[feature_names].values)
        labels.append(slice_df["attack"].values.astype(int))
        n_extracted += 1
        if max_sequences_to_extract is not None and n_extracted >= max_sequences_to_extract:
            break

    if n_extracted == 0:
        logger("No sequences extracted.")
        return None

    dataset = DeviceSequenceDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers,
                      pin_memory=bool(num_workers > 0 and torch.cuda.is_available()))
