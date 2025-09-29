# Utils that mirror the old helpers but use the new config system

import os
from pathlib import Path
from privateer_ad.config import PathConfig

def get_dataset_path_old(name: str) -> str:
    """
    Map 'train'|'val'|'test' to new processed dir, identical filenames.
    """
    pc = PathConfig()
    return str(pc.processed_dir.joinpath(f"{name}.csv"))

def check_existing_datasets_old() -> None:
    """
    Prevent accidental overwrite (old behavior).
    """
    for mode in ["train", "val", "test"]:
        p = Path(get_dataset_path_old(mode))
        if p.exists():
            raise FileExistsError(f"File {p} exists.")
