from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
import yaml

@dataclass
class DeviceInfo:
    imeisv: str
    ip: str
    type: str
    malicious: bool
    in_attacks: List[int]  # keep as int list, cast from yaml if needed

@dataclass
class AttackInfo:
    start: str
    stop: str

@dataclass
class FeatureInfo:
    dtype: str = "str"
    drop: bool = False
    is_input: bool = False

class OldMetaData:
    def __init__(self, metadata_path: str | Path | None = None):
        self._path = Path(metadata_path or Path(__file__).with_name("old_metadata.yaml"))
        with open(self._path, "r") as f:
            data = yaml.safe_load(f)

        # coerce types same as old project
        self.devices: Dict[str, DeviceInfo] = {
            k: DeviceInfo(
                imeisv=str(v["imeisv"]),
                ip=v["ip"],
                type=v["type"],
                malicious=bool(v["malicious"]),
                in_attacks=[int(x) for x in v["in_attacks"]],
            )
            for k, v in data["devices"].items()
        }
        self.attacks: Dict[str, AttackInfo] = {
            str(k): AttackInfo(**v) for k, v in data["attacks"].items()
        }
        self.features: Dict[str, FeatureInfo] = {
            k: FeatureInfo(
                dtype=v.get("dtype", "str"),
                drop=bool(v.get("drop", False)),
                is_input=bool(v.get("is_input", False)),
            )
            for k, v in data["features"].items()
        }

    def get_input_features(self) -> List[str]:
        return [f for f, info in self.features.items() if info.is_input]

    def get_drop_features(self) -> List[str]:
        return [f for f, info in self.features.items() if info.drop]
