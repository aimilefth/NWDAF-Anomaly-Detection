import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

from privateer_ad.config import PathConfig

@dataclass
class DownloadConfig:
    zip_name: str = "nwdaf-data.zip"
    url: str = PathConfig().data_url
    extraction_dir: Path = PathConfig().raw_dir
    raw_dataset: Path = PathConfig().raw_dataset

class Downloader:
    def __init__(self, config: DownloadConfig):
        self.url = config.url
        self.extraction_dir = config.extraction_dir
        self.zip_path = self.extraction_dir.joinpath(config.zip_name)

    def download(self):
        print(f"[old_data_utils] Downloading from {self.url} ...")
        resp = requests.get(self.url, stream=True)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        self.extraction_dir.mkdir(parents=True, exist_ok=True)
        print(f"[old_data_utils] Saving to {self.zip_path} ...")
        with open(self.zip_path, "wb") as f, tqdm(
            total=total, unit="iB", unit_scale=True, unit_divisor=1024
        ) as pbar:
            for chunk in resp.iter_content(chunk_size=1024):
                size = f.write(chunk)
                pbar.update(size)

    def extract(self):
        print(f"[old_data_utils] Extracting to {self.extraction_dir} ...")
        with zipfile.ZipFile(self.zip_path, "r") as zf:
            files = zf.namelist()
            for file in tqdm(files, desc="Extracting"):
                zf.extract(file, self.extraction_dir)
        print("[old_data_utils] Extract completed.")

    def remove_zip(self):
        try:
            print(f"[old_data_utils] Removing {self.zip_path} ...")
            os.remove(self.zip_path)
        except FileNotFoundError:
            pass

    def download_extract(self):
        self.download()
        self.extract()
        self.remove_zip()
