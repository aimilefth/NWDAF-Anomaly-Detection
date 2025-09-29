"""
Old data processing logic, adapted to the new config system.

Key behaviors preserved:
- Per-device split with stratification on 'attack_number'
- Re-label 'attack' per device using metadata.in_attacks (if desired)
- Scale using StandardScaler fit on benign only (attack_number==0, or attack==0 fallback)
- Optional PCA fitted on benign only
- Dataloaders via PyTorch Forecasting with time index per device (imeisv)
- Old defaults: seq_len=12, large batch size by default (caller can override)
"""

import os
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from pandas import DataFrame

from datasets import Dataset
from pytorch_forecasting import TimeSeriesDataSet
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from flwr_datasets.partitioner import PathologicalPartitioner
from torch.utils.data import DataLoader

from privateer_ad.config import PathConfig, MetadataConfig
from privateer_ad.old_data_utils.utils import (
    get_dataset_path_old,
    check_existing_datasets_old,
)


class OldDataProcessor:
    """Old pipeline entry point (paths/metadata from new config)."""

    def __init__(self):
        self.paths = PathConfig()
        self.metadata = MetadataConfig()
        self.input_features = self.metadata.get_input_features()
        self.drop_features = self.metadata.get_drop_features()
        self.devices = self.metadata.devices  # dict[str, DeviceInfo]

        self.scaler_path = self.paths.scalers_dir.joinpath("scaler.pkl")
        self.pca_path = self.paths.scalers_dir.joinpath("pca.pkl")
        self.scaler: Optional[StandardScaler] = None
        self.pca: Optional[PCA] = None
        self.partitioner: Optional[PathologicalPartitioner] = None

    # ------------------------
    # High-level orchestration
    # ------------------------
    def initialize_data_pipeline(
        self,
        dataset_path: str | Path | None = None,
        train_size: float = 0.8,
    ) -> None:
        """
        Old behavior: split raw CSV per device with stratification, save train/val/test,
        and fit scaler/PCA on train benign only.
        """
        dataset_path = Path(dataset_path) if dataset_path else self.paths.raw_dataset
        check_existing_datasets_old()

        raw_df = pd.read_csv(dataset_path)
        print(f"[old_data_utils] Raw shape: {raw_df.shape}")
        self.split_data(raw_df, train_size=train_size)
        # Fit scalers on train set
        self.preprocess_data(
            path="train",
            setup=True,  # fit scaler and pca on benign train
            use_pca=False,  # only fit; caller can apply PCA later if needed
        )

    # -------------
    # Core pieces
    # -------------
    def split_data(self, df: DataFrame, train_size=0.8) -> None:
        """
        Old behavior:
          - For each device, (optionally) recompute 'attack' using metadata.in_attacks
          - Stratify on 'attack_number' when splitting
          - Save train/val/test in processed/ as CSV
        """
        train_dfs, val_dfs, test_dfs = [], [], []

        # ensure imeisv comparable type with metadata
        df["imeisv"] = df["imeisv"].astype(str)

        for dev_key, dev_info in self.devices.items():
            device_df = df.loc[df["imeisv"] == dev_info.imeisv].copy()

            # If the raw file doesn't already have consistent 'attack' per device,
            # re-derive it using metadata.in_attacks (old logic).
            if "attack_number" in device_df.columns:
                device_df.loc[
                    device_df["attack_number"].isin(dev_info.in_attacks), "attack"
                ] = 1
                device_df.loc[
                    ~device_df["attack_number"].isin(dev_info.in_attacks), "attack"
                ] = 0

            if device_df.empty:
                continue

            # stratify on 'attack_number' if available; otherwise on 'attack'
            strat_col = "attack_number" if "attack_number" in device_df.columns else "attack"

            df_train, df_tmp = train_test_split(
                device_df,
                train_size=train_size,
                stratify=device_df[strat_col],
                random_state=42,
            )
            # Split val/test evenly out of the remaining
            df_val, df_test = train_test_split(
                df_tmp, test_size=0.5, stratify=df_tmp[strat_col], random_state=42
            )
            train_dfs.append(df_train)
            val_dfs.append(df_val)
            test_dfs.append(df_test)

        df_train = pd.concat(train_dfs).sort_values("_time").reset_index(drop=True)
        df_val = pd.concat(val_dfs).sort_values("_time").reset_index(drop=True)
        df_test = pd.concat(test_dfs).sort_values("_time").reset_index(drop=True)

        self.paths.processed_dir.mkdir(parents=True, exist_ok=True)
        print("[old_data_utils] Saving split datasets ...")
        df_train.to_csv(get_dataset_path_old("train"), index=False)
        df_val.to_csv(get_dataset_path_old("val"), index=False)
        df_test.to_csv(get_dataset_path_old("test"), index=False)

    def clean_data(self, df: DataFrame) -> DataFrame:
        df = df.drop(columns=self.drop_features, errors="ignore")
        df = df.drop_duplicates()
        # Handle missing times similarly to new code (be cautious)
        if "_time" in df.columns and df["_time"].isna().any():
            df = df.dropna(subset=["_time"])
        df = df.dropna(axis="rows")
        return df.reset_index(drop=True)

    # -------------
    # Scaling / PCA
    # -------------
    def _benign_mask(self, df: DataFrame):
        if "attack_number" in df.columns:
            return df["attack_number"] == 0
        if "attack" in df.columns:
            return df["attack"] == 0
        # fallback: use all
        return pd.Series([True] * len(df), index=df.index)

    def setup_scaler(self, df: DataFrame) -> StandardScaler:
        benign_df = df[self._benign_mask(df)].copy()
        self.paths.scalers_dir.mkdir(parents=True, exist_ok=True)
        scaler = StandardScaler().fit(benign_df[self.input_features])
        joblib.dump(scaler, self.scaler_path)
        self.scaler = scaler
        print(f"[old_data_utils] Scaler saved at {self.scaler_path}")
        return scaler

    def load_scaler(self) -> StandardScaler:
        if self.scaler is None:
            self.scaler = joblib.load(self.scaler_path)
        return self.scaler

    def apply_scale(self, df: DataFrame) -> DataFrame:
        scaler = self.load_scaler()
        df.loc[:, self.input_features] = scaler.transform(df[self.input_features])
        return df

    def setup_pca(self, df: DataFrame, n_components: int = 10) -> PCA:
        benign_df = df[self._benign_mask(df)].copy()
        self.paths.scalers_dir.mkdir(parents=True, exist_ok=True)
        pca = PCA(n_components=n_components).fit(benign_df[self.input_features])
        joblib.dump(pca, self.pca_path)
        self.pca = pca
        print(f"[old_data_utils] PCA saved at {self.pca_path}")
        return pca

    def load_pca(self) -> PCA:
        if self.pca is None:
            self.pca = joblib.load(self.pca_path)
        return self.pca

    def apply_pca(self, df: DataFrame) -> DataFrame:
        pca = self.load_pca()
        X = pca.transform(df[self.input_features])
        cols = pca.get_feature_names_out().tolist()
        proj = pd.DataFrame(X, columns=cols, index=df.index)
        df = df.drop(columns=self.input_features)
        return pd.concat([df, proj], axis=1)

    # ----------------
    # Preprocess entry
    # ----------------
    def preprocess_data(
        self,
        path: str | Path,
        use_pca: bool = False,
        n_components: Optional[int] = None,
        partition_id: Optional[int] = None,
        setup: bool = False,
    ) -> DataFrame:
        """
        Old path that:
          - reads 'train'|'val'|'test' CSV (or a path)
          - optional partition (Pathological, by imeisv)
          - clean -> (fit scaler/PCA if setup) -> scale -> (apply PCA) -> sort by time
        """
        # Resolve path
        if isinstance(path, str) and path in ("train", "val", "test"):
            path = get_dataset_path_old(path)
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        df = pd.read_csv(path, low_memory=False)
        # enforce str imeisv to match metadata keys
        df["imeisv"] = df["imeisv"].astype(str)

        if partition_id is not None:
            df = self.get_partition(df, partition_id, num_partitions=len(self.devices))

        df = self.clean_data(df)

        if setup:
            # Fit on this dataset (train benign only)
            self.setup_scaler(df)
            if n_components is not None:
                self.setup_pca(df, n_components=n_components)

        df = self.apply_scale(df)

        if use_pca:
            if n_components is not None and setup:
                # already fit above
                pass
            df = self.apply_pca(df)

        return df.sort_values("_time").reset_index(drop=True)

    # -----------
    # Partitions
    # -----------
    def get_partition(self, df: DataFrame, partition_id: int, num_partitions: int) -> DataFrame:
        """
        Old approach: non-IID partitions by 'imeisv' using PathologicalPartitioner.
        """
        if self.partitioner is None:
            self.partitioner = PathologicalPartitioner(
                num_partitions=num_partitions,
                num_classes_per_partition=1,
                partition_by="imeisv",
                class_assignment_mode="random",
                shuffle=False,
            )
            self.partitioner.dataset = Dataset.from_pandas(df)
        part_df = self.partitioner.load_partition(partition_id).to_pandas(batched=False)
        return part_df[df.columns]

    # -----------
    # Dataloaders
    # -----------
    def get_dataloader(
        self,
        split: str,               # 'train'|'val'|'test'
        use_pca: bool = False,
        batch_size: int = 4096,   # old default large batch by default
        seq_len: int = 12,        # old default seq_len
        partition_id: Optional[int] = None,
        only_benign: bool = False,
        num_workers: int = 16,
        shuffle: bool = True,
    ) -> DataLoader:
        """
        Old dataloader with per-device time index (imeisv), sequence modeling.
        """
        df = self.preprocess_data(
            path=split,
            use_pca=use_pca,
            setup=False,
            partition_id=partition_id,
        )

        if only_benign:
            if "attack" in df.columns:
                df = df[df["attack"] == 0]
            else:
                print("[old_data_utils] WARNING: 'attack' column missing; benign filter skipped.")

        df = df.sort_values(by=["_time"]).reset_index(drop=True)
        if use_pca:
            input_cols = [c for c in df.columns if c.startswith("pca")]
        else:
            input_cols = self.input_features

        # per-device time index as in old code
        df["time_idx"] = df.groupby("imeisv")["_time"].cumcount()

        ts = TimeSeriesDataSet(
            data=df,
            time_idx="time_idx",
            target="attack",
            group_ids=["imeisv"],
            max_encoder_length=seq_len,
            max_prediction_length=1,
            time_varying_unknown_reals=input_cols,
            scalers=None,
            target_normalizer=None,
            allow_missing_timesteps=False,
        )

        return ts.to_dataloader(
            train=(split == "train"),
            batch_size=batch_size,
            num_workers=min(os.cpu_count() or 1, num_workers),
            pin_memory=True,
            prefetch_factor=20 if num_workers > 0 else None,
            persistent_workers=bool(num_workers > 0),
            shuffle=shuffle if split == "train" else False,
        )
