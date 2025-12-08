"""
PRIVATEER Federated Learning Model Integrated Dashboard
"""
import time
import threading
import queue
import logging
import os
import hashlib
import sys
from collections import deque
from pathlib import Path

from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import torch
from privateer_filter_inference.inference_dense import predict
from urllib3.exceptions import InsecureRequestWarning
import urllib3

from dash import dcc, html, Input, Output, State

from privateer_ad.architectures.transformer_ad import TransformerAD
from privateer_ad.etl import DataProcessor
from privateer_ad.config import (
    DataConfig,
    MetadataConfig,
    ModelConfig,
    PathConfig,
    TrainingConfig,
    MLFlowConfig
)
from privateer_ad.utils import load_model_weights, load_mlflow_model_from_run
try:
    from privateer_ad.alveo.alveo_runner import AlveoRunner, AlveoRunnerParameters
    ALVEO_MODULE_AVAILABLE = True
except ImportError:
    logging.warning("⚠️ AlveoRunner module not found in 'privateer_ad.alveo'. FPGA features will be disabled.")
    AlveoRunner = None
    AlveoRunnerParameters = None
    ALVEO_MODULE_AVAILABLE = False


exported_anomalies = []

SHAP_MAX_SEQ_LEN = int(os.getenv('SHAP_MAX_SEQ_LEN', '12'))

DEFAULT_ANON_EPSILON = float(os.getenv('ANONYMIZER_EPSILON', '0.0'))
ANONYMIZER_SENSITIVITY = float(os.getenv('ANONYMIZER_SENSITIVITY', '0.0'))
EPSILON_MIN = float(os.getenv('ANONYMIZER_EPSILON_MIN', '0.01'))
EPSILON_MAX = float(os.getenv('ANONYMIZER_EPSILON_MAX', '1.0'))
EPSILON_STEP = float(os.getenv('ANONYMIZER_EPSILON_STEP', '0.01'))
SENSITIVE_FEATURES = tuple(os.getenv('ANONYMIZER_SENSITIVE_FEATURES', 'dl_bitrate,ul_bitrate').split(','))
# EXPERIMENT_MODEL_ID = os.getenv('PRIVATEER_EXPERIMENT_ID', 'experiments/20250313-181907')
# This contains the model 
# [adv_trained_model_on_anonymized_data.zip](https://spacecollab.sharepoint.com/:u:/r/sites/PRIVATEER/Shared%20Documents/WP3.%20Decentralised%20Robust%20Security%20Analytics/Anomaly%20Detection%20Model/adv_trained_model_on_anonymized_data.zip?csf=1&web=1&e=PwV1f7)
EXPERIMENT_MODEL_ID = os.getenv('PRIVATEER_EXPERIMENT_ID', 'experiments/adv_anonymized_model/adv_trained_model_attack_eps_0.005.pt')

def _env_as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {'1', 'true', 'yes', 'on'}

def _env_as_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if not value:
        return None
    return value

MISP_BASE_URL = os.getenv('MISP_URL') or "https://10.160.3.60"
MISP_API_KEY = os.getenv('MISP_API_KEY') or "QJVfx7A4SwMY7UsgNtzNE47Qtv5D4Qf0cBAPSCdr"
MISP_VERIFY_SSL = _env_as_bool(os.getenv('MISP_VERIFY_SSL'), default=False)
MISP_TIMEOUT = float(os.getenv('MISP_TIMEOUT', '5.0'))
MISP_USERNAME = os.getenv('MISP_USERNAME') or "infili@misp.testing"
MISP_PASSWORD = os.getenv('MISP_PASSWORD') or "]3L2>9bzS'RFM*Z"

MLFLOW_RUN_ID = _env_as_str('PRIVATEER_MLFLOW_RUN_ID')
MLFLOW_RUN_NAME = _env_as_str('PRIVATEER_MLFLOW_RUN_NAME', 'bright-chimp-326')
MLFLOW_ARTIFACT_PATH = _env_as_str('PRIVATEER_MLFLOW_ARTIFACT_PATH', 'TransformerAD')
# I dont have val_inference.csv.
INFERENCE_DATASET_FILENAME = os.getenv('PRIVATEER_INFERENCE_DATASET', 'val_inference.csv')
TUNING_DATASET_FILENAME = os.getenv('PRIVATEER_TUNING_DATASET', 'val.csv')
RECOMPUTE_THRESHOLD = _env_as_bool(os.getenv('PRIVATEER_RECOMPUTE_THRESHOLD'), default=False)
TARGET_FPR = float(os.getenv('PRIVATEER_TARGET_FPR', '0.01'))
THRESHOLD_STRATEGY = os.getenv('PRIVATEER_THRESHOLD_STRATEGY', 'f1').strip().lower()
TARGET_PRECISION = float(os.getenv('PRIVATEER_TARGET_PRECISION', '0.90'))

USE_FPGA = _env_as_bool(os.getenv('PRIVATEER_USE_FPGA'), default=True)
ALVEO_XCLBIN_PATH = os.getenv('PRIVATEER_ALVEO_XCLBIN_PATH', os.path.join('experiments', "alveo_xclbins", "attention_ae_adv_anon_W16A16.xclbin"))

urllib3.disable_warnings(InsecureRequestWarning)


XAI_SHAP_BASE_URL = "http://localhost:5000/xai/shap"

# Simulation pacing: default to 0s (no artificial delay) so throughput reflects raw model speed.
SIMULATION_INTERVAL_SECONDS = float(os.getenv('PRIVATEER_SIM_INTERVAL', '0.0'))


def _shap_iframe_src(view: str, serial: int) -> str:
    return f"{XAI_SHAP_BASE_URL}/{view}?refresh={serial}"

SHAP_IFRAME_STYLE = {
    "width": "1400px",
    "height": "1000px",
    "border": "0",
    "transform": "scale(0.6)",
    "transformOrigin": "0 0"
}

SHAP_CONTAINER_STYLE = {
    "width": "840px",
    "height": "600px",
    "margin": "0 auto",
    "overflow": "hidden"
}

FEATURE_DISPLAY_NAMES = {
    'dl_bitrate': 'DL Rate',
    'ul_bitrate': 'UL Rate'
}

FEATURE_COLORS = {
    'dl_bitrate': '#0d6efd',
    'ul_bitrate': '#198754'
}

RAW_TRACE_COLOR = '#6c757d'


def _build_shap_iframe(view: str, serial: int) -> html.Iframe:
    """Construct an iframe pointing to the given SHAP view."""
    return html.Iframe(
        id=f"shap-{view}-frame",
        src=_shap_iframe_src(view, serial),
        style=SHAP_IFRAME_STYLE
    )


def _shap_placeholder(message: str = "no anomalies detected") -> html.Div:
    """Create a placeholder element shown when SHAP data is unavailable."""
    return html.Div(
        message,
        className="text-center text-muted fw-bold",
        style={
            "display": "flex",
            "alignItems": "center",
            "justifyContent": "center",
            "height": "100%",
            "width": "100%",
            "backgroundColor": "#f8f9fa",
            "border": "1px dashed #ced4da",
            "borderRadius": "8px"
        }
    )


iframe_refresh_serial = 0
REALTIME_INTERVAL_MS = 500  # Base interval for standard dashboard updates (milliseconds)
XAI_INTERVAL_MS = 5000      # XAI refresh cadence (milliseconds)
XAI_WINDOW_SECONDS = 5
THROUGHPUT_WINDOW_SECONDS = 60  # Fixed aggregation window for throughput (seconds)
last_xai_anomaly_timestamp = None


class MISPClient:
    """Lightweight client for pushing anomaly events into MISP."""

    def __init__(self,
                 base_url: str | None = MISP_BASE_URL,
                 api_key: str | None = MISP_API_KEY,
                 verify_ssl: bool = MISP_VERIFY_SSL,
                 timeout: float = MISP_TIMEOUT):
        self.base_url = base_url.rstrip('/') if base_url else None
        self.api_key = api_key
        self.verify_ssl = verify_ssl
        self.timeout = timeout
        self._warned_unconfigured = False

    def _configured(self) -> bool:
        return bool(self.base_url and self.api_key)

    def publish_anomaly(self,
                        *,
                        ip: str,
                        detection_time: datetime,
                        device_id: str,
                        reconstruction_error: float,
                        threshold: float) -> None:
        if not self._configured():
            if not self._warned_unconfigured:
                logging.info("MISP client not configured; skipping event publication.")
                self._warned_unconfigured = True
            return

        event_payload = {
            "Event": {
                "info": f"PRIVATEER anomaly detected for device {device_id}",
                "distribution": 0,
                "threat_level_id": 3,
                "analysis": 0,
                "Attribute": [
                    {
                        "category": "Network activity",
                        "type": "ip-dst",
                        "value": ip,
                        "to_ids": True,
                        "comment": "Device IP observed during anomaly detection."
                    },
                    {
                        "category": "Other",
                        "type": "text",
                        "value": detection_time.isoformat(),
                        "to_ids": False,
                        "comment": "Detection timestamp (UTC)."
                    },
                    {
                        "category": "Other",
                        "type": "text",
                        "value": f"reconstruction_error={reconstruction_error:.6f}",
                        "to_ids": False,
                        "comment": f"Model threshold at detection time: {threshold:.6f}"
                    }
                ]
            }
        }

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": self.api_key
        }

        try:
            response = requests.post(
                f"{self.base_url}/events/add",
                headers=headers,
                json=event_payload,
                timeout=self.timeout,
                verify=self.verify_ssl
            )
            if not response.ok:
                logging.warning(
                    "Failed to publish MISP event (status %s): %s",
                    response.status_code,
                    response.text
                )
        except Exception as exc:
            logging.warning("Unable to publish anomaly to MISP: %s", exc)


class DemoAnonymizer:
    """Apply lightweight anonymization and noise injection for demo samples."""

    def __init__(self,
                 epsilon: float = DEFAULT_ANON_EPSILON,
                 sensitivity: float = ANONYMIZER_SENSITIVITY,
                 sensitive_features: tuple[str, ...] = SENSITIVE_FEATURES):
        self.epsilon = max(float(epsilon), 1e-6)
        self.sensitivity = max(float(sensitivity), 1e-6)
        # Normalize feature names to strip whitespace and keep consistent casing
        self.sensitive_features = tuple(name.strip() for name in sensitive_features if name.strip())
        self._rng = np.random.default_rng()

    def update_epsilon(self, new_epsilon: float) -> None:
        """Update epsilon used by the Laplace mechanism."""
        self.epsilon = max(float(new_epsilon), 1e-6)

    def anonymize(self, device_id: str | None, feature_values: dict[str, float]) -> dict:
        """Hash the device identifier and obfuscate sensitive feature values."""
        #hashed_device = self._hash_device(device_id)
        obfuscated_features: dict[str, float] = {}
        quality_loss: dict[str, float] = {}
        scale = self.sensitivity / self.epsilon

        for feature, value in feature_values.items():
            if feature in self.sensitive_features and value is not None:
                noisy_value = float(value + self._rng.laplace(0.0, scale))
                obfuscated_features[feature] = noisy_value
                quality_loss[feature] = abs(noisy_value - float(value))
            else:
                obfuscated_features[feature] = value

        return {
            #'anonymized_device_id': hashed_device,
            'anonymized_device_id': device_id,
            'feature_values': obfuscated_features,
            'quality_loss': quality_loss
        }

    @staticmethod
    def _hash_device(device_id: str | None) -> str:
        if device_id is None:
            return "anon-0000"
        digest = hashlib.sha256(str(device_id).encode('utf-8')).hexdigest()
        return f"anon-{int(digest[:8], 16) % 10000:04d}"


class PrivateerAnomalyDetector:
    """Core anomaly detection engine using TransformerAD with differential privacy."""

    def __init__(
        self,
        device_override: str | torch.device | None = None,
        model_name: str = 'TransformerAD_DP',
        experiment_id: str | None = None,
        mlflow_run_name: str | None = MLFLOW_RUN_NAME,
        mlflow_run_id: str | None = MLFLOW_RUN_ID,
        mlflow_artifact_path: str | None = MLFLOW_ARTIFACT_PATH,
        runner: AlveoRunner = None,
    ):
        """Initialize detector with specified model and privacy configurations."""
        self.model_name = model_name
        self.experiment_id = experiment_id or EXPERIMENT_MODEL_ID
        self.mlflow_run_name = mlflow_run_name
        self.mlflow_run_id = mlflow_run_id
        self.mlflow_artifact_path = mlflow_artifact_path or 'global_TransformerAD'
        self.runner = runner

        self.data_config = DataConfig()
        self.data_config.num_workers = 0
        self.data_config.pin_memory = False
        self.data_config.batch_size = 1
        self.data_config.prefetch_factor = None
        self.data_config.persistent_workers = False
        self.data_config.seq_len = SHAP_MAX_SEQ_LEN
        self.metadata = MetadataConfig()
        self.input_features = self.metadata.get_input_features()
        self.mlflow_config = MLFlowConfig()
        self.paths_config = PathConfig()
        self.inference_dataset_path = (self.paths_config.processed_dir / INFERENCE_DATASET_FILENAME).as_posix()
        self.tuning_dataset_path = (self.paths_config.processed_dir / TUNING_DATASET_FILENAME).as_posix()

        if device_override is not None:
            self.device = torch.device(device_override)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize DataProcessor with streaming config
        self.data_processor = DataProcessor(self.data_config)
        self.test_ds = self.data_processor.get_dataset(self.inference_dataset_path, only_benign=False)
        # Shuffle for demo to surface attacks quickly
        self.test_dl = self.data_processor.get_dataloader(self.inference_dataset_path, only_benign=False, train=True)
        self.threshold = 0.0209596287459135  # Default threshold
        self.loss_fn = None
        self.model = None
        self.hitl_enabled = False

        if not self._load_model_from_mlflow():
            self._load_local_model()

        if RECOMPUTE_THRESHOLD:
            try:
                self.recompute_threshold(target_fpr=TARGET_FPR)
            except Exception as exc:
                logging.warning("Unable to recompute threshold: %s", exc)

        logging.info(f"input features: {self.input_features}")

    def _load_model_from_mlflow(self) -> bool:
        """Attempt to pull the model + metadata from MLflow."""
        if not (self.mlflow_run_id or self.mlflow_run_name):
            return False

        tracking_candidates: list[str] = []
        if self.mlflow_config.tracking_uri:
            tracking_candidates.append(self.mlflow_config.tracking_uri)

        local_tracking = f"file://{PathConfig().root_dir / 'mlruns'}"
        if local_tracking not in tracking_candidates:
            tracking_candidates.append(local_tracking)

        for tracking_uri in tracking_candidates:
            try:
                model, threshold, loss_fn, run = load_mlflow_model_from_run(
                    tracking_uri=tracking_uri,
                    run_id=self.mlflow_run_id,
                    run_name=self.mlflow_run_name,
                    artifact_path=self.mlflow_artifact_path,
                    experiment_name=self.mlflow_config.experiment_name,
                )
                self.model = model.to(self.device)
                self.model.eval()
                self.threshold = float(threshold)
                self.loss_fn = loss_fn
                run_label = run.data.tags.get('mlflow.runName') or self.mlflow_run_name or self.mlflow_run_id
                self.model_name = run_label or self.model_name
                self.experiment_id = f"mlflow:{run.info.experiment_id}/{run.info.run_id}"
                logging.info(
                    "Loaded MLflow model '%s' (run_id=%s) from %s with threshold %.6f",
                    run.data.tags.get('mlflow.runName') or self.mlflow_run_name or self.mlflow_run_id,
                    run.info.run_id,
                    tracking_uri,
                    self.threshold,
                )
                return True
            except Exception as exc:
                logging.warning(
                    "Failed to load MLflow run %s (tracking_uri=%s): %s",
                    self.mlflow_run_name or self.mlflow_run_id,
                    tracking_uri,
                    exc
                )
        return False

    def _load_local_model(self) -> None:
        """Fallback to loading weights from the local experiments directory."""
        logging.info(f"Falling back to experiment artifacts at {self.experiment_id}...")
        model_config = ModelConfig(
            model_name=self.model_name,
            input_size=len(self.input_features),
            seq_len=SHAP_MAX_SEQ_LEN
        )
        paths_config = PathConfig()
        state_dict = load_model_weights(self.experiment_id, paths_config)
        self.model = TransformerAD(model_config=model_config)
        load_result = self.model.load_state_dict(state_dict, strict=False)
        if isinstance(load_result, tuple):
            missing_keys, unexpected_keys = load_result
        else:
            missing_keys = getattr(load_result, 'missing_keys', [])
            unexpected_keys = getattr(load_result, 'unexpected_keys', [])
        if missing_keys or unexpected_keys:
            logging.warning(
                "Model state dict mismatch; missing keys: %s, unexpected keys: %s",
                missing_keys,
                unexpected_keys
            )
        self.model.to(self.device)
        self.model.eval()
        loss_fn_name = TrainingConfig().loss_fn_name
        self.loss_fn = getattr(torch.nn, loss_fn_name)(reduction='none')

    def detect_anomaly(self, input_batch):
        """
        Run anomaly detection on input batch.

        Returns:
            tuple: (is_anomaly, reconstruction_error, true_label)
        """
        try:
            # Extract input tensor and true label
            input_tensor = input_batch[0]['encoder_cont'].to(self.device)
            true_label = input_batch[1][0].item() if len(input_batch) > 1 else None

            # Run inference
            with torch.no_grad():
                output = self.model(input_tensor)
                reconstruction_error = self.loss_fn(input_tensor, output).mean(dim=(1, 2)).item()

                # Determine if anomaly
                is_anomaly = reconstruction_error > self.threshold

                if self.hitl_enabled and is_anomaly:
                    try:
                        hitl_result = predict(input_tensor.cpu().numpy())
                        if hitl_result and isinstance(hitl_result, list):
                            is_anomaly = bool(hitl_result[0].get('is_anomaly', is_anomaly))
                    except Exception as exc:
                        logging.warning("HITL prediction failed: %s", exc)

                return is_anomaly, reconstruction_error, true_label

        except Exception as e:
            logging.error(f"❌ Error in anomaly detection: {e}")
            return False, 0.0, None

    def detect_anomaly_alveo(self, input_batch):
        """
        Run anomaly detection using the Alveo FPGA Runner.
        """
        try:
            # Extract input tensor and true label
            # Shape is expected to be [1, seq_len, features]
            input_tensor = input_batch[0]['encoder_cont']
            true_label = input_batch[1][0].item() if len(input_batch) > 1 else None
            
            # Convert to numpy for Alveo
            input_np = input_tensor.cpu().numpy()
            
            # Run inference on FPGA
            # run_vector expects flattened input, output_shape handles the reshape back
            output_np = self.runner.run_vector(input_np, output_shape=input_np.shape)
            
            # Calculate reconstruction error (L1 / Mean Absolute Error)
            # Matching logic: self.loss_fn(input, output).mean(dim=(1, 2))
            diff = np.abs(input_np - output_np)
            reconstruction_error = np.mean(diff)

            # Determine if anomaly
            is_anomaly = reconstruction_error > self.threshold

            return is_anomaly, float(reconstruction_error), true_label

        except Exception as e:
            logging.error(f"❌ Error in Alveo anomaly detection: {e}")
            return False, 0.0, None

    def update_threshold(self, new_threshold):
        """Update anomaly detection threshold for real-time adjustment."""
        self.threshold = new_threshold
        logging.info(f"🎯 Threshold updated to: {new_threshold:.6f}")

    def recompute_threshold(self, target_fpr: float = 0.01) -> float:
        """
        Recompute threshold using the current model on the inference dataset.

        Strategy:
            1) Run the current model over the configured tuning dataset.
            2) If labels are available (0/1) and contain both classes, choose a threshold
               using the requested strategy:
                   - precision (default): highest threshold with precision >= TARGET_PRECISION
                     (ties broken by higher recall), else best F1.
                   - f1: threshold that maximizes F1.
                   - youden: maximize (tpr - fpr).
            3) Otherwise, fall back to the (1 - target_fpr) quantile of benign samples;
               if no benign labels exist, use the 99th percentile of all.
        """
        tuning_path = self.tuning_dataset_path
        if not Path(tuning_path).exists():
            logging.warning("Tuning dataset %s not found; falling back to inference dataset %s", tuning_path, self.inference_dataset_path)
            tuning_path = self.inference_dataset_path

        eval_dl = self.data_processor.get_dataloader(tuning_path, only_benign=False, train=False)
        errors: list[float] = []
        labels: list[int | None] = []

        self.model.eval()
        with torch.no_grad():
            for batch in eval_dl:
                input_tensor = batch[0]['encoder_cont'].to(self.device)
                label = batch[1][0].item() if len(batch) > 1 else None
                output = self.model(input_tensor)
                err = self.loss_fn(input_tensor, output).mean(dim=(1, 2)).item()
                errors.append(float(err))
                labels.append(int(label) if label is not None else None)

        labeled_pairs = [(e, l) for e, l in zip(errors, labels) if l is not None]
        has_pos = any(l == 1 for _, l in labeled_pairs)
        has_neg = any(l == 0 for _, l in labeled_pairs)

        if labeled_pairs and has_pos and has_neg:
            errs = np.array([p[0] for p in labeled_pairs], dtype=float)
            labs = np.array([p[1] for p in labeled_pairs], dtype=int)
            sort_idx = np.argsort(errs)
            errs = errs[sort_idx]
            labs = labs[sort_idx]
            thresholds = np.unique(errs)

            best_cutoff = self.threshold
            best_info = {'strategy': 'f1', 'score': -1.0, 'tp': 0, 'fp': 0, 'tn': 0, 'fn': 0}

            def _eval(thr):
                preds = errs > thr
                tp = int(((preds == 1) & (labs == 1)).sum())
                fp = int(((preds == 1) & (labs == 0)).sum())
                tn = int(((preds == 0) & (labs == 0)).sum())
                fn = int(((preds == 0) & (labs == 1)).sum())
                precision = tp / (tp + fp) if tp + fp > 0 else 0.0
                recall = tp / (tp + fn) if tp + fn > 0 else 0.0
                fpr = fp / (fp + tn) if fp + tn > 0 else 0.0
                denom = (precision + recall)
                f1 = 0.0 if denom == 0 else 2 * precision * recall / denom
                return dict(threshold=float(thr), tp=tp, fp=fp, tn=tn, fn=fn,
                            precision=precision, recall=recall, f1=f1, fpr=fpr)

            # Evaluate all unique thresholds (vector loop)
            evaluations = [_eval(thr) for thr in thresholds]

            chosen = None
            if THRESHOLD_STRATEGY == 'precision':
                feasible = [e for e in evaluations if e['precision'] >= TARGET_PRECISION and e['recall'] > 0]
                if feasible:
                    feasible.sort(key=lambda e: (e['threshold'], e['recall']), reverse=True)
                    chosen = feasible[0]
            elif THRESHOLD_STRATEGY == 'youden':
                evaluations.sort(key=lambda e: (e['recall'] - e['fpr'], e['recall']), reverse=True)
                chosen = evaluations[0]

            if chosen is None:
                evaluations.sort(key=lambda e: e['f1'], reverse=True)
                chosen = evaluations[0]

            cutoff = chosen['threshold']
            best_info = chosen | {'strategy': THRESHOLD_STRATEGY if chosen in evaluations else 'f1-fallback'}
            source = (
                f"{best_info['strategy']} "
                f"(thr={best_info['threshold']:.6f}, f1={best_info['f1']:.3f}, "
                f"prec={best_info['precision']:.3f}, rec={best_info['recall']:.3f}, "
                f"tp={best_info['tp']}, fp={best_info['fp']}, tn={best_info['tn']}, fn={best_info['fn']})"
            )
        else:
            benign_errors = [e for e, l in zip(errors, labels) if l == 0]
            if benign_errors:
                cutoff = np.percentile(benign_errors, 100 * (1 - target_fpr))
                source = "benign-only percentile"
            else:
                cutoff = np.percentile(errors, 99)
                source = "all-samples fallback"

        self.threshold = float(cutoff)
        logging.info(
            "🔧 Recomputed threshold=%.6f using %d samples (%s, target_fpr=%.3f, target_precision=%.3f, strategy=%s, attacks=%d, normals=%d) [tuning_ds=%s]",
            self.threshold,
            len(errors),
            source,
            target_fpr,
            TARGET_PRECISION,
            THRESHOLD_STRATEGY,
            sum(1 for l in labels if l == 1),
            sum(1 for l in labels if l == 0),
            tuning_path,
        )
        return self.threshold


class NetworkTrafficSimulator:
    """Simulates network traffic using real dataset for demonstration purposes."""
    def __init__(self, detector, anonymizer, device_label: str | None = None):
        """Initialize simulator with anomaly detector and anonymization logic."""
        self.detector = detector
        self.anonymizer = anonymizer
        self.device_label = device_label or ('fpga' if self.detector.device.type == 'cuda' else 'cpu')
        self.data_queue = queue.Queue()
        self.running = False
        self.thread = None
        self.current_sample_index = 0
        self.dataloader_iterator = iter(self.detector.test_dl)
        self._fallback_devices = list(self.detector.metadata.devices.keys()) or ['default']
        self.misp_client = MISPClient()
        self.latency_window = deque(maxlen=1000)

    def reset_iterator(self):
        """Reset dataloader to beginning for continuous simulation."""
        self.dataloader_iterator = iter(self.detector.test_dl)
        self.current_sample_index = 0
        logging.info("🔄 Dataloader iterator reset to beginning")

    def get_next_sample(self):
        """Fetch next sample from dataset, cycling back when exhausted."""
        try:
            sample = next(self.dataloader_iterator)
            self.current_sample_index += 1
            return sample
        except StopIteration:
            # End of dataset, restart from beginning
            logging.warning("📄 End of dataset reached, restarting from beginning")
            self.reset_iterator()
            return self.get_next_sample()

    def start_simulation(self, interval=0.1):
        """Begin traffic simulation in separate thread with specified interval."""
        if self.running:
            logging.debug("Simulation already running for %s, skipping start", self.device_label)
            return
        self.running = True
        self.thread = threading.Thread(target=self._simulation_loop, args=(interval,))
        self.thread.daemon = True
        self.thread.start()
        logging.info(f"▶️ Simulation started on {self.device_label} with {interval}s interval")

    def stop_simulation(self):
        """Stop simulation and clean up thread resources."""
        self.running = False
        if self.thread:
            self.thread.join()
        logging.warning("⏸️ Simulation stopped")

    def _simulation_loop(self, interval):
        """Main simulation execution loop running in background thread."""
        while self.running:
            try:
                # Get next sample from dataloader
                sample = self.get_next_sample()

                # Detect anomaly and record latency
                start = time.perf_counter()
                is_anomaly, score, true_label = self.detector.detect_anomaly(sample)
                inference_latency_ms = (time.perf_counter() - start) * 1000

                is_anomaly_alveo = None
                score_alveo = None
                true_label_alveo = None
                latency_ms_alveo = None

                if self.detector.runner is not None:
                    start_alveo = time.perf_counter()
                    is_anomaly_alveo, score_alveo, true_label_alveo = self.detector.detect_anomaly_alveo(sample)
                    latency_ms_alveo = (time.perf_counter() - start_alveo) * 1000

                self.latency_window.append(inference_latency_ms)
                avg_latency = (sum(self.latency_window) / len(self.latency_window)
                               if self.latency_window else inference_latency_ms)
                logging.info(
                    "⏱️ Detection latency: %.2f ms (avg %.2f ms over %d samples)",
                    inference_latency_ms,
                    avg_latency,
                    len(self.latency_window)
                )

                # Create result dictionary
                result = {
                    'timestamp': datetime.now(),
                    'sample_index': self.current_sample_index,
                    'runtime_device': self.device_label,
                    'is_anomaly': is_anomaly,
                    'reconstruction_error': score,
                    'true_label': true_label,
                    # Alveo specific data
                    'is_anomaly_alveo': is_anomaly_alveo,
                    'score_alveo': score_alveo,
                    'true_label_alveo': true_label_alveo,
                    'latency_ms_alveo': latency_ms_alveo,
                    # Common data
                    'input_tensor': sample[0]['encoder_cont'].cpu().numpy(),
                    'latency_ms': inference_latency_ms,
                    'feature_values': {},
                    'shap': None
                }

                # Extract feature values for display
                input_flat = sample[0]['encoder_cont'].squeeze().cpu().numpy()
                if len(input_flat.shape) == 2:  # [seq_len, features]
                    # Take the last timestep for current values
                    current_features = input_flat[-1]
                    for i, feature_name in enumerate(self.detector.input_features):
                        if i < len(current_features):
                            result['feature_values'][feature_name] = float(current_features[i])

                device_id = None
                group_key = None
                if hasattr(self.detector.test_ds, 'group_ids'):
                    for candidate in ('imeisv', 'device_id', 'device'):
                        if candidate in self.detector.test_ds.group_ids:
                            group_key = candidate
                            break

                groups_tensor = sample[0].get("groups") if isinstance(sample[0], dict) else None
                if group_key and groups_tensor is not None:
                    try:
                        device_id = self.detector.test_ds.transform_values(
                            group_key,
                            groups_tensor,
                            inverse=True,
                            group_id=True
                        )
                        if hasattr(device_id, 'item'):
                            device_id = device_id.item()
                    except KeyError:
                        logging.debug("Group key %s not found in dataset transformers", group_key)
                    except Exception as e:
                        logging.debug("Unable to extract device ID using key %s: %s", group_key, e)

                if not device_id:
                    fallback_idx = (self.current_sample_index - 1) % len(self._fallback_devices)
                    device_id = self._fallback_devices[fallback_idx]

                device_id = str(device_id)
                device_info = self.detector.metadata.devices.get(device_id)
                ip = device_info.ip if device_info else device_id

                shap_payload = None

                if is_anomaly:
                    shap_payload = self._calculate_shap(sample[0]['encoder_cont'])
                    info_misp = {
                        'ip': ip,
                        'time': result['timestamp']
                    }
                    print("Anomaly detected, info_misp:", info_misp)
                    exported_anomalies.append(info_misp)
                    self.misp_client.publish_anomaly(
                        ip=ip,
                        detection_time=result['timestamp'],
                        device_id=device_id,
                        reconstruction_error=score,
                        threshold=self.detector.threshold
                    )

                if shap_payload:
                    result['shap'] = shap_payload

                # Apply anonymization logic for identifiers and features
                result['raw_feature_values'] = dict(result['feature_values']) 
                anonymized_payload = self.anonymizer.anonymize(ip, result['feature_values'])
                result.update(anonymized_payload)
                result['raw_device_id'] = device_id
                # Put result in queue
                self.data_queue.put(result)

                if interval > 0:
                    time.sleep(interval)

            except Exception as e:
                logging.error(f"❌ Error in simulation loop: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(interval)

    def get_latest_data(self):
        """Retrieve all pending simulation results from queue."""
        data = []
        while not self.data_queue.empty():
            try:
                data.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return data

    def _calculate_shap(self, tensor: torch.Tensor):
        """Call XAI backend to obtain SHAP values for the provided tensor."""
        try:
            model_cfg = getattr(self.detector.model, 'model_config', None)
            shap_seq_len = None
            if model_cfg is not None:
                shap_seq_len = getattr(model_cfg, 'seq_len', None)
            if shap_seq_len is None:
                shap_seq_len = tensor.shape[1]

            shap_seq_len = max(1, min(shap_seq_len, tensor.shape[1], SHAP_MAX_SEQ_LEN))
            if tensor.shape[1] > shap_seq_len:
                tensor = tensor[:, :shap_seq_len, :]
            tensor_cpu = tensor.detach().to(torch.float32).cpu()
            payload = {
                'data': tensor_cpu.tolist(),
                'shape': list(tensor_cpu.shape),
                'dtype': str(tensor_cpu.dtype)
            }
            response = requests.post(
                'http://localhost:5000/api/shap/calculate/string_json',
                json=payload,
                timeout=10
            )
            if response.status_code != 200:
                try:
                    detail = response.text
                except Exception:
                    detail = '<no response body>'
                logging.warning(f"SHAP request failed with status {response.status_code}: {detail}")
                return None
            shap_json = response.json()
            contribution = []
            if isinstance(shap_json, dict):
                if isinstance(shap_json.get('both'), dict):
                    contribution = shap_json['both'].get('contribution', [])
                else:
                    contribution = shap_json.get('contribution', [])
            return {
                'shap_values': shap_json.get('shap_values', {}),
                'contribution': contribution,
                'feature_names': shap_json.get('feature_names', []),
            }
        except Exception as e:
            logging.warning(f"Unable to retrieve SHAP values: {e}")
            return None


def init_alveo_runner(xclbin_path: str, device_num: int, seq_len: int, n_features: int): 
    # Requirement: Run the setup source logic before initializing pynq
    # source ${SETUP_DIR}/${SETUP_FILE}
    if not ALVEO_MODULE_AVAILABLE:
        return None

    alveo_runner = None
    try:
        import pynq # Import here to ensure env vars (XILINX_XRT) are set
        
        # Check for XCLBIN        
        if os.path.exists(xclbin_path):
            try:
                device_list = list(pynq.Device.devices)
            except Exception as e:
                logging.warning(f"⚠️ Failed to list PYNQ devices (XRT issues?): {e}")
                device_list = []

            if device_list:
                # Ensure index is within bounds
                if device_num >= len(device_list):
                    logging.warning(f"⚠️ Device index {device_num} out of bounds. Using 0.")
                    device_num = 0
                
                device = device_list[device_num]
                logging.info(f"🔮 Found Alveo Device: {device.name}")
                
                params = AlveoRunnerParameters(
                    input_buffer_elements=seq_len * n_features,
                    output_buffer_elements=seq_len * n_features,
                    kernel_name=None 
                )
                
                alveo_runner = AlveoRunner(
                    bitstream_path=xclbin_path,
                    parameters=params,
                    device=device,
                    device_bus=None
                )
                logging.info("✅ AlveoRunner initialized.")
            else:
                logging.warning("⚠️ No PYNQ devices found. Alveo acceleration disabled.")
        else:
            logging.warning(f"⚠️ XCLBIN not found at {xclbin_path}. Alveo acceleration disabled.")
        return alveo_runner
    except ImportError:
        logging.warning("⚠️ PYNQ library not installed. Alveo acceleration disabled.")
        return None
    except Exception as e:
        logging.warning(f"⚠️ AlveoRunner init failed: {e}")
        return None


# Initialize components
logging.info("🔄 Initializing PRIVATEER components...")
if USE_FPGA:
    alveo_runner = init_alveo_runner(xclbin_path=ALVEO_XCLBIN_PATH, device_num=0, seq_len=12, n_features=8)
else:
    alveo_runner = None

detector = PrivateerAnomalyDetector(runner=alveo_runner)
anonymizer = DemoAnonymizer()
simulator = NetworkTrafficSimulator(detector, anonymizer)
simulators = [simulator]

# Spin up a parallel CPU simulator when GPU is available so we can benchmark both
cpu_detector = None
cpu_simulator = None
if torch.cuda.is_available() and detector.device.type != 'cpu':
    cpu_detector = PrivateerAnomalyDetector(device_override='cpu')
    cpu_simulator = NetworkTrafficSimulator(cpu_detector, anonymizer, device_label='cpu')
    simulators.append(cpu_simulator)
    logging.info("Enabling dual-device simulation: gpu + cpu")
else:
    logging.info("Running single-device simulation on %s", detector.device.type)

# Storage for real-time data
realtime_data = {
    'timestamp': [],
    'sample_index': [],
    'runtime_device': [],
    'reconstruction_error': [],
    'is_anomaly': [],
    'true_label': [],
    'anonymized_device_id': [],
    'latency_ms': [],
    # --- ALVEO ---
    'is_anomaly_alveo': [],
    'score_alveo': [],
    'true_label_alveo': [],
    'latency_ms_alveo': [],
    # -------------
    'raw_feature_values': {},  
    'feature_values': {},
    'shap_values': [],
    'raw_inputs': []
}

# Initialize feature storage
for feature in detector.input_features:
    realtime_data['feature_values'][feature] = []
    realtime_data['raw_feature_values'][feature] = [] 

max_points = 200  # Keep last 200 points for display
min_threshold = float(np.floor(detector.threshold * .1))
max_threshold = float(np.ceil(detector.threshold * 10.))
step_threshold = 0.0001


def _empty_hitl_store():
    """Default HITL store structure with queue support."""
    return {
        'payload': None,
        'pending': False,
        'anomaly_id': None,
        'feedback': None,
        'queue': [],
        'last_feedback': None,
        'dataset_index': 0
    }


def _load_hitl_samples():
    """Load samples from the validation inference CSV for HITL review."""
    primary_path = PathConfig().processed_dir / INFERENCE_DATASET_FILENAME
    repo_root = Path(__file__).resolve().parent.parent
    alt_path = repo_root.parent / 'data' / 'processed' / INFERENCE_DATASET_FILENAME

    path = primary_path
    if not path.exists() and alt_path.exists():
        path = alt_path

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        logging.warning("Unable to load HITL dataset at %s: %s", path, exc)
        return []

    samples = []
    for idx, row in df.iterrows():
        features = {}
        for feat in detector.input_features:
            if feat in row:
                try:
                    features[feat] = float(row[feat])
                except Exception:
                    features[feat] = row[feat]
        ts = row.get('timestamp') if 'timestamp' in row else None
        display_ts = ts if isinstance(ts, str) else f"Sample #{idx + 1}"
        samples.append({
            'timestamp': ts if isinstance(ts, str) else display_ts,
            'display_timestamp': display_ts,
            'features': features,
            'anomaly_number': idx + 1
        })
    logging.info("Loaded %d HITL samples from %s", len(samples), path)
    return samples


HITL_SAMPLES = _load_hitl_samples()


def _next_hitl_payload(dataset_index: int):
    """Return the next HITL payload from the preloaded dataset."""
    if dataset_index < len(HITL_SAMPLES):
        payload = dict(HITL_SAMPLES[dataset_index])
        if 'anomaly_number' not in payload:
            payload['anomaly_number'] = dataset_index + 1
        return payload
    return None


def build_hitl_panel(anomaly_data, feedback_message, last_feedback=None):
    """Render the HITL feedback panel for the most recent anomaly."""
    if not isinstance(anomaly_data, dict):
        anomaly_data = {}

    available = bool(anomaly_data)
    timestamp_text = anomaly_data.get('display_timestamp') or "Waiting for anomalies"
    features = anomaly_data.get('features', {}) if available else {}
    if not isinstance(features, dict):
        features = {}
    anomaly_number = anomaly_data.get('anomaly_number') if available else None

    feature_items = []
    if features:
        for name, value in list(features.items())[:10]:
            try:
                value_text = f"{float(value):.3f}"
            except (TypeError, ValueError):
                value_text = str(value)
            feature_items.append(html.Li(f"{name}: {value_text}", className="mb-1"))
    else:
        feature_items.append(html.Li("Feature details unavailable", className="text-muted"))

    feedback_text = "Awaiting feedback" if available else "Waiting for anomalies"
    if feedback_message:
        feedback_text = feedback_message
    elif last_feedback:
        feedback_text = f"Last submission: {last_feedback}"

    badge = None
    if feedback_message:
        color = "success" if "True" in feedback_message else "danger"
        badge = dbc.Badge(feedback_message, color=color, className="ms-2")
    elif last_feedback:
        badge = dbc.Badge(last_feedback, color="secondary", className="ms-2")

    return dbc.Card(
        [
            dbc.CardBody([
                html.Div([
                    html.Div([
                        html.H5("Detected Anomaly", className="mb-1"),
                        html.Small(
                            f"Anomaly #{anomaly_number}" if anomaly_number is not None else "",
                            className="text-primary d-block"
                        ),
                        html.Small(timestamp_text, className="text-muted"),
                        html.Div(badge, className="mt-1")
                    ]),
                    html.Div([
                        dbc.Button(
                            "🧠 Enable HITL",
                            id="enable-hitl-btn",
                            color="info",
                            outline=True,
                            className="mb-1",
                            n_clicks=0
                        ),
                        html.Small(id="hitl-enable-status", className="text-muted")
                    ], className="d-flex flex-column align-items-end"),
                    html.Div([
                        dbc.Button(
                            "True Anomaly",
                            id="hitl-true-btn",
                            color="success",
                            outline=True,
                            className="mb-2",
                            disabled=not available,
                            style={"width": "160px", "fontWeight": "600"}
                        ),
                        dbc.Button(
                            "False Anomaly",
                            id="hitl-false-btn",
                            color="danger",
                            outline=True,
                            disabled=not available,
                            style={"width": "160px", "fontWeight": "600"}
                        )
                    ], className="d-flex flex-column align-items-start align-items-md-end gap-2")
                ], className="d-md-flex justify-content-between align-items-start"),
                html.Hr(),
                html.Div([
                    html.Strong("Features:", className="d-block mb-2"),
                    html.Ul(feature_items, className="mb-0")
                ]),
                html.Div(feedback_text, className="mt-3 text-muted")
            ])
        ],
        style={"border": "2px solid #6fa8dc", "borderRadius": "10px"}
    )


initial_hitl_panel = build_hitl_panel(None, None, None)

# Create Dash app
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True
)
app.title = "PRIVATEER - Federated Learning Model Integrated Dashboard"

# App layout
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col([
            html.H1("🛡️ PRIVATEER Federated Learning Model Integrated Dashboard",
                    className="text-center mb-4"),
            html.P("Privacy-Preserving Anomaly Detection for 6G Networks",
                   className="text-center text-muted"),
            html.Hr(),
        ])
    ]),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("Runtime Controls", className="card-title"),
                    dbc.ButtonGroup([
                        dbc.Button("▶️ Start Simulation", id="start-btn", color="success", className="me-2"),
                        dbc.Button("⏸️ Stop Simulation", id="stop-btn", color="danger", className="me-2"),
                        dbc.Button("🔄 Reset", id="reset-btn", color="warning")
                    ]),
                    html.Hr(),
                    html.Div([
                        html.Label("🎯 Anomaly Threshold:", className="form-label"),
                        dcc.Slider(
                            id='threshold-slider',
                            min=min_threshold,
                            max=max_threshold,
                            step=step_threshold,
                            value=detector.threshold,
                            marks={
                                value: f"{value:.3f}"
                                for value in np.linspace(min_threshold, max_threshold, 10)
                            },                            tooltip={"placement": "bottom", "always_visible": True}
                        )
                    ], className="mb-3"),
                    html.Div([
                        html.Label("🔏 Privacy ε (Laplace Noise):", className="form-label"),
                        dcc.Slider(
                            id='epsilon-slider',
                            min=EPSILON_MIN,
                            max=EPSILON_MAX,
                            step=EPSILON_STEP,
                            value=anonymizer.epsilon,
                            marks={
                                float(f"{value:.2f}"): f"{value:.2f}"
                                for value in np.linspace(EPSILON_MIN, EPSILON_MAX, 5)
                            },
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Small(f"Current ε: {anonymizer.epsilon:.2f}", id='epsilon-display', className="text-muted")
                    ], className="mb-3"),
                    html.Div([
                        html.Label("🔐 Privacy Protection: ", className="form-label"),
                        dbc.Badge("Anonymization Active", color="success", className="ms-2"),
                        html.Small(" - Device IDs are anonymized", className="text-muted ms-2")
                    ]),
                    html.Div(id="status-indicator", className="mt-3")
                ])
            ])
        ], width=12)
    ], className="mb-4"),

dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 Network Feature Values (Privacy-Preserved)", className="card-title"),
                    # Changed height to 850px to cover the two stacked graphs on the right
                    dcc.Graph(id="feature-display", style={'height': '850px'})
                ])
            ])
        ], width=6),

        dbc.Col([
            # CPU Detection
            dbc.Card([
                dbc.CardBody([
                    html.H4("🚨 Anomaly Detection Results (CPU)", className="card-title"),
                    dcc.Graph(id="anomaly-detection-cpu", style={'height': '400px'})
                ])
            ], className="mb-4"),
            # FPGA Detection (Stacked below CPU)
            dbc.Card([
                dbc.CardBody([
                    html.H4("🚨 Anomaly Detection Results (FPGA)", className="card-title"),
                    dcc.Graph(id="anomaly-detection-fpga", style={'height': '400px'})
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🧠 Expert Feedback (HITL)", className="card-title"),
                    html.Div(id="hitl-panel", className="mt-3", children=initial_hitl_panel)
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📉 Real-Time False Positive Rate", className="card-title"),
                    dcc.Graph(id="fpr-trend", style={'height': '600px'})
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🪟 Window Contribution to Decision (Last anomaly detected)", className="card-title"),
                    html.Div(
                        id="shap-window-container",
                        children=_shap_placeholder(),
                        style=SHAP_CONTAINER_STYLE
                    )
                ])
            ])
        ], width=6),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📈 Feature Contribution to Decision (Last anomaly detected)", className="card-title"),
                    html.Div(
                        id="shap-features-container",
                        children=_shap_placeholder(),
                        style=SHAP_CONTAINER_STYLE
                    )
                ])
            ])
        ], width=6)
    ], className="mb-4"),

    dbc.Row([
        # Width changed from 6 to 4
        dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4("⚡ Inference Latency (CPU vs FPGA)", className="card-title"),
                            html.Small(
                                f"Average per-sample inference latency over the last {THROUGHPUT_WINDOW_SECONDS} seconds.",
                                className="text-muted d-block mb-3"
                            ),
                            dcc.Input(
                                id='throughput-window-seconds',
                                type='number',
                                value=THROUGHPUT_WINDOW_SECONDS,
                                style={'display': 'none'},
                                readOnly=True
                            ),
                            dcc.Graph(
                                id="latency-graph",
                                style={'height': '520px', 'width': '100%', 'margin': '0 auto'}
                            )
                        ])
            ])
        ], width=4),
        
        # NEW COLUMN for Absolute Difference
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🔍 Reconstruction Error Difference", className="card-title"),
                    html.Small(
                        "Absolute difference between CPU and FPGA reconstruction errors.",
                        className="text-muted d-block mb-3"
                    ),
                    dcc.Graph(
                        id="diff-graph",
                        style={'height': '520px', 'width': '100%', 'margin': '0 auto'}
                    )
                ])
            ])
        ], width=4),

        # Width changed from 6 to 4
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📊 Feature Influence SHAP", className="card-title"),
                    html.Div(
                        id="shap-timeseries-container",
                        children=_shap_placeholder(),
                        style=SHAP_CONTAINER_STYLE
                    )
                ])
            ])
        ], width=4)
    ], className="mb-4", justify="center"),

    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("📈 Detection Statistics", className="card-title"),
                    html.Div(id="stats-display")
                ])
            ])
        ], width=8),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H4("🔒 Anonymized Devices", className="card-title"),
                    html.Div(id="device-list", style={'max-height': '200px', 'overflow-y': 'auto'})
                ])
            ])
        ], width=4)
    ]),

    # dcc.Interval(
    #     id='interval-component',
    #     interval=3000,  # Deprecated: original three-second refresh
    #     n_intervals=0,
    #     disabled=True
    # ),
    dcc.Interval(
        id='realtime-interval',
        interval=REALTIME_INTERVAL_MS,
        n_intervals=0,
        disabled=True
    ),
    dcc.Interval(
        id='xai-interval',
        interval=XAI_INTERVAL_MS,
        n_intervals=0,
        disabled=True
    ),

    # Store simulation state
    dcc.Store(id='simulation-state', data={'running': False}),
    dcc.Store(id='hitl-store', data=_empty_hitl_store()),
    dcc.Store(id='hitl-enabled', data=False)

], fluid=True)


# Callbacks
@app.callback(
    Output('simulation-state', 'data', allow_duplicate=True),
    Input('threshold-slider', 'value'),
    prevent_initial_call=True
)
def update_threshold(threshold):
    """Callback to handle threshold slider changes in real-time."""
    for sim in simulators:
        sim.detector.update_threshold(threshold)
    return dash.no_update


@app.callback(
    [Output('epsilon-display', 'children'),
     Output('simulation-state', 'data', allow_duplicate=True)],
    Input('epsilon-slider', 'value'),
    prevent_initial_call=True
)
def update_epsilon(epsilon):
    """Handle epsilon slider updates for anonymization noise."""
    anonymizer.update_epsilon(epsilon)
    return f"Current ε: {anonymizer.epsilon:.2f}", dash.no_update

def create_status_badge(text, color):
    """Generate status indicator with current simulation state."""
    device_labels = ", ".join(sorted({sim.device_label for sim in simulators}))
    sample_total = sum(sim.current_sample_index for sim in simulators)
    return dbc.Row([
        dbc.Col([
            dbc.Badge(f"Status: {text}", color=color, className="fs-6 me-2"),
            dbc.Badge(f"Model: {detector.model_name} [{detector.experiment_id}]", color="info", className="fs-6"),
            dbc.Badge(f"Devices: {device_labels}", color="secondary", className="fs-6 ms-2"),
            dbc.Badge(f"Samples: {sample_total}", color="secondary", className="fs-6 ms-2"),
            dbc.Badge(f"ε: {anonymizer.epsilon:.3f}", color="warning", className="fs-6 ms-2")
        ])
    ])


def collect_latest_data():
    """Gather pending samples from all active simulators."""
    aggregated = []
    for sim in simulators:
        aggregated.extend(sim.get_latest_data())
    return aggregated


@app.callback(
    [Output('hitl-enabled', 'data'),
     Output('enable-hitl-btn', 'children'),
     Output('enable-hitl-btn', 'disabled'),
     Output('enable-hitl-btn', 'color'),
     Output('hitl-enable-status', 'children'),
     Output('hitl-store', 'data', allow_duplicate=True)],
    Input('enable-hitl-btn', 'n_clicks'),
    State('hitl-enabled', 'data'),
    State('hitl-store', 'data'),
    prevent_initial_call=True
)
def enable_hitl(n_clicks, already_enabled, hitl_data):
    """Enable HITL inference path, seed the first CSV anomaly, and lock the control."""
    if not n_clicks or already_enabled:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update, dash.no_update

    for sim in simulators:
        sim.detector.hitl_enabled = True

    store = _empty_hitl_store()
    if isinstance(hitl_data, dict):
        for key in store:
            if key == 'queue':
                store[key] = list(hitl_data.get(key) or [])
            else:
                store[key] = hitl_data.get(key, store[key])

    if not store.get('payload'):
        first_payload = _next_hitl_payload(0)
        if first_payload:
            store.update(
                payload=first_payload,
                anomaly_id=first_payload.get('anomaly_number'),
                pending=True,
                feedback=None,
                queue=[],
                dataset_index=1
            )

    status_text = f"HITL enabled — reviewing anomalies from {INFERENCE_DATASET_FILENAME}."
    return True, "HITL Enabled", True, "secondary", status_text, store


@app.callback(
    [Output('simulation-state', 'data'),
     Output('realtime-interval', 'disabled'),
     Output('xai-interval', 'disabled'),
     Output('status-indicator', 'children')],
    [Input('start-btn', 'n_clicks'),
     Input('stop-btn', 'n_clicks'),
     Input('reset-btn', 'n_clicks')],
    [State('simulation-state', 'data')]
)
def control_simulation(start_clicks, stop_clicks, reset_clicks, state):
    """Handle simulation control buttons and state management."""
    ctx = dash.callback_context
    global last_xai_anomaly_timestamp

    if not ctx.triggered:
        return state, True, True, create_status_badge("Stopped", "danger")

    button_id = ctx.triggered[0]['prop_id'].split('.')[0]

    if button_id == 'start-btn' and start_clicks:
        for sim in simulators:
            sim.start_simulation(interval=SIMULATION_INTERVAL_SECONDS)
        return {'running': True}, False, False, create_status_badge("Running", "success")

    elif button_id == 'stop-btn' and stop_clicks:
        for sim in simulators:
            sim.stop_simulation()
        last_xai_anomaly_timestamp = None
        return {'running': False}, True, True, create_status_badge("Stopped", "danger")

    elif button_id == 'reset-btn' and reset_clicks:
        for sim in simulators:
            sim.stop_simulation()
        last_xai_anomaly_timestamp = None
        # Clear realtime data
        for key in realtime_data:
            if key not in ['feature_values', 'anonymized_device_id']:
                realtime_data[key].clear()
            elif key == 'feature_values':
                for feature in realtime_data['feature_values']:
                    realtime_data['feature_values'][feature].clear()
            elif key == 'anonymized_device_id':
                realtime_data[key].clear()
        # Reset dataloader iterator
        for sim in simulators:
            sim.reset_iterator()
        return {'running': False}, True, True, create_status_badge("Reset", "warning")

    return state, True, True, create_status_badge("Stopped", "danger")


@app.callback(
    [Output('feature-display', 'figure'),
     Output('anomaly-detection-cpu', 'figure'),   # Renamed from anomaly-detection
     Output('anomaly-detection-fpga', 'figure'),  # NEW
     Output('diff-graph', 'figure'),              # NEW
     Output('fpr-trend', 'figure'),
     Output('stats-display', 'children'),
     Output('device-list', 'children'),
     Output('latency-graph', 'figure')],
    [Input('realtime-interval', 'n_intervals'),
     Input('throughput-window-seconds', 'value')],
    [State('simulation-state', 'data')]
)
def update_graphs(n, throughput_window, state):
    """Main callback for updating all dashboard visualizations."""
    if not state.get('running', False):
        return (
            create_empty_figure("Simulation Stopped"),
            create_empty_figure("Simulation Stopped"),
            create_empty_figure("Simulation Stopped"),
            create_empty_figure("Simulation Stopped"),
            create_empty_figure("Simulation Stopped"),
            html.P("Start simulation to see statistics"),
            html.P("No devices detected yet"),
            create_empty_figure("Simulation Stopped")
        )

    # Get new data
    new_data = collect_latest_data()

    # Add new data to realtime storage
    for data_point in new_data:
        realtime_data['timestamp'].append(data_point['timestamp'])
        realtime_data['sample_index'].append(data_point['sample_index'])
        realtime_data['runtime_device'].append(data_point.get('runtime_device', detector.device.type))
        realtime_data['reconstruction_error'].append(data_point['reconstruction_error'])
        realtime_data['is_anomaly'].append(data_point['is_anomaly'])
        realtime_data['true_label'].append(data_point['true_label'])
        realtime_data['anonymized_device_id'].append(data_point['anonymized_device_id'])
        realtime_data['raw_inputs'].append(data_point['input_tensor'])
        realtime_data['latency_ms'].append(data_point.get('latency_ms'))
        # --- ALVEO ---
        realtime_data['is_anomaly_alveo'].append(data_point.get('is_anomaly_alveo'))
        realtime_data['score_alveo'].append(data_point.get('score_alveo'))
        realtime_data['true_label_alveo'].append(data_point.get('true_label_alveo'))
        realtime_data['latency_ms_alveo'].append(data_point.get('latency_ms_alveo'))
        # -------------
        # Add feature values (anonymized)
        for feature, value in data_point['feature_values'].items():
            if feature in realtime_data['feature_values']:
                realtime_data['feature_values'][feature].append(value)

        # Add feature values (raw) with fallback reconstruction from input tensor
        raw_fp = data_point.get('raw_feature_values')
        if not raw_fp:
            tensor = data_point.get('input_tensor')
            raw_fp = {}
            if tensor is not None:
                tensor_np = np.squeeze(np.array(tensor))
                if tensor_np.ndim == 2:
                    last_step = tensor_np[-1]
                    for idx, feature in enumerate(detector.input_features):
                        if idx < len(last_step):
                            raw_fp[feature] = float(last_step[idx])
        if raw_fp:
            for feature, value in raw_fp.items():
                if feature in realtime_data['raw_feature_values']:
                    realtime_data['raw_feature_values'][feature].append(value)
        
        if data_point.get('shap'):
            realtime_data['shap_values'].append(data_point['shap'])
        else:
            realtime_data['shap_values'].append(None)

    # Limit data size
    if len(realtime_data['timestamp']) > max_points:
        for key in realtime_data:
            if key == 'feature_values':
                for feature in realtime_data['feature_values']:
                    realtime_data['feature_values'][feature] = realtime_data['feature_values'][feature][-max_points:]
            elif key == 'raw_feature_values':   # <-- ADD THIS ELIF
                for feature in realtime_data['raw_feature_values']:
                    realtime_data['raw_feature_values'][feature] = realtime_data['raw_feature_values'][feature][-max_points:]
            elif key in ('shap_values', 'raw_inputs', 'latency_ms'):
                realtime_data[key] = realtime_data[key][-max_points:]
            else:
                realtime_data[key] = realtime_data[key][-max_points:]

    feature_fig = create_feature_figure()
    # Create two separate anomaly figures
    anomaly_fig_cpu = create_anomaly_figure(source='cpu')
    anomaly_fig_fpga = create_anomaly_figure(source='alveo')
    # Create the difference figure
    diff_fig = create_diff_figure()

    fpr_fig = create_fpr_figure()
    stats = create_statistics()
    device_list = create_device_list()
    latency_fig = create_latency_figure(throughput_window or THROUGHPUT_WINDOW_SECONDS)

    return (
        feature_fig,
        anomaly_fig_cpu,
        anomaly_fig_fpga,
        diff_fig,
        fpr_fig,
        stats,
        device_list,
        latency_fig
    )

@app.callback(
    Output('hitl-panel', 'children'),
    Input('hitl-store', 'data'),
    prevent_initial_call=False
)
def render_hitl_panel(hitl_data):
    """Render HITL panel reflecting current pending anomaly and feedback state."""
    try:
        if not hitl_data:
            return build_hitl_panel(None, None, None)
        return build_hitl_panel(
            hitl_data.get('payload'),
            hitl_data.get('feedback'),
            hitl_data.get('last_feedback')
        )
    except Exception as exc:
        logging.warning("Unable to render HITL panel: %s", exc)
        return build_hitl_panel(None, None, None)


@app.callback(
    Output('hitl-store', 'data', allow_duplicate=True),
    [Input('hitl-true-btn', 'n_clicks'),
     Input('hitl-false-btn', 'n_clicks')],
    State('hitl-store', 'data'),
    prevent_initial_call=True
)
def record_hitl_feedback(true_clicks, false_clicks, hitl_data):
    """Mark the current anomaly as reviewed and store dummy feedback."""
    if not hitl_data or not hitl_data.get('pending'):
        return dash.no_update

    ctx = dash.callback_context
    if not ctx.triggered:
        return dash.no_update

    trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    feedback_message = None
    if trigger == 'hitl-true-btn':
        feedback_message = "Submitted feedback: True Anomaly"
    elif trigger == 'hitl-false-btn':
        feedback_message = "Submitted feedback: False Anomaly"
    else:
        return dash.no_update

    base_store = _empty_hitl_store()
    if isinstance(hitl_data, dict):
        for key in base_store:
            if key == 'queue':
                base_store[key] = list(hitl_data.get(key) or [])
            else:
                base_store[key] = hitl_data.get(key, base_store[key])

    queue = list(base_store.get('queue') or [])
    dataset_index = int(base_store.get('dataset_index') or 0)
    updated = dict(base_store)
    updated['last_feedback'] = feedback_message
    updated['feedback'] = feedback_message
    updated['pending'] = False

    if queue:
        next_payload = queue.pop(0)
        updated.update(
            payload=next_payload,
            anomaly_id=next_payload.get('anomaly_number'),
            pending=True,
            feedback=None,
            queue=queue,
            dataset_index=dataset_index
        )
    else:
        next_payload = _next_hitl_payload(dataset_index)
        if next_payload:
            updated.update(
                payload=next_payload,
                anomaly_id=next_payload.get('anomaly_number'),
                pending=True,
                feedback=None,
                queue=[],
                dataset_index=dataset_index + 1
            )
        else:
            updated.update(payload=None, anomaly_id=None, queue=[], pending=False, dataset_index=dataset_index)

    return updated


@app.callback(
    [Output('shap-timeseries-container', 'children'),
     Output('shap-window-container', 'children'),
     Output('shap-features-container', 'children')],
    [Input('xai-interval', 'n_intervals')],
    [State('simulation-state', 'data')]
)
def update_xai_sections(n, state):
    """Refresh XAI panels at a slower cadence, focusing on the latest anomalies."""
    global iframe_refresh_serial, last_xai_anomaly_timestamp
    if not state.get('running', False):
        last_xai_anomaly_timestamp = None
        return _shap_placeholder(), _shap_placeholder(), _shap_placeholder()

    cutoff_time = datetime.now() - timedelta(seconds=XAI_WINDOW_SECONDS)
    target_index = None
    fallback_index = None

    # Walk data history from newest to oldest until we find the latest anomaly.
    for idx in range(len(realtime_data['timestamp']) - 1, -1, -1):
        if not realtime_data['is_anomaly'][idx]:
            continue

        sample_time = realtime_data['timestamp'][idx]
        if sample_time >= cutoff_time:
            target_index = idx
            break

        if fallback_index is None:
            fallback_index = idx

    if target_index is None:
        target_index = fallback_index

    if target_index is None:
        last_xai_anomaly_timestamp = None
        return _shap_placeholder(), _shap_placeholder(), _shap_placeholder()

    anomaly_timestamp = realtime_data['timestamp'][target_index]
    shap_payload = realtime_data['shap_values'][target_index]

    # Lazily fetch SHAP data the first time we reference this anomaly.
    if shap_payload is None:
        raw_input = realtime_data['raw_inputs'][target_index]
        try:
            tensor = torch.tensor(raw_input, dtype=torch.float32)
            shap_payload = simulator._calculate_shap(tensor)
            if shap_payload:
                realtime_data['shap_values'][target_index] = shap_payload
            else:
                logging.warning("SHAP backend returned empty payload for anomaly at %s", anomaly_timestamp)
        except Exception as exc:
            logging.warning("Failed to fetch SHAP data for anomaly at %s: %s", anomaly_timestamp, exc)
            shap_payload = None

    if not shap_payload:
        last_xai_anomaly_timestamp = None
        return _shap_placeholder(), _shap_placeholder(), _shap_placeholder()

    if last_xai_anomaly_timestamp != anomaly_timestamp:
        iframe_refresh_serial += 1
        last_xai_anomaly_timestamp = anomaly_timestamp

    iframe_timeseries = _build_shap_iframe("timeseries", iframe_refresh_serial)
    iframe_window = _build_shap_iframe("window", iframe_refresh_serial)
    iframe_features = _build_shap_iframe("features", iframe_refresh_serial)
    return iframe_timeseries, iframe_window, iframe_features


# Add this NEW callback just for updating the sample counter
@app.callback(
    Output('status-indicator', 'children', allow_duplicate=True),
    [Input('realtime-interval', 'n_intervals')],
    [State('simulation-state', 'data')],
    prevent_initial_call=True
)
def update_sample_counter(n, state):
    """Update sample counter display during active simulation."""
    if state.get('running', False):
        return create_status_badge("Running", "success")
    return dash.no_update


def create_empty_figure(title):
    """Generate placeholder figure when no data is available."""
    fig = go.Figure()
    fig.add_annotation(
        text=title,
        xref="paper", yref="paper",
        x=0.5, y=0.5, xanchor='center', yanchor='middle',
        showarrow=False, font=dict(size=20)
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor='white'
    )
    return fig


def create_throughput_figure(window_seconds):
    """Show average samples per second each device processed in the latest aggregation window."""
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    try:
        window_seconds = float(window_seconds)
    except (TypeError, ValueError):
        window_seconds = THROUGHPUT_WINDOW_SECONDS
    if window_seconds <= 0:
        window_seconds = THROUGHPUT_WINDOW_SECONDS

    latest_ts = realtime_data['timestamp'][-1]
    cutoff = latest_ts - timedelta(seconds=window_seconds)

    counts_by_device: dict[str, int] = {}
    for ts, device_label in zip(realtime_data['timestamp'], realtime_data.get('runtime_device', [])):
        if ts < cutoff:
            continue
        label = (device_label or 'cpu').upper()
        if label == 'GPU':
            label = 'FPGA'
        counts_by_device[label] = counts_by_device.get(label, 0) + 1

    if not counts_by_device or window_seconds == 0:
        return create_empty_figure("No Data Available")

    samples_per_second = {
        label: count / window_seconds for label, count in counts_by_device.items()
    }

    fig = go.Figure()
    labels = sorted(samples_per_second.keys())
    fig.add_trace(go.Bar(
        x=labels,
        y=[samples_per_second[label] for label in labels],
        marker=dict(color=['#7480ff' if lbl == 'CPU' else '#ff8b73' for lbl in labels])
    ))

    fig.update_layout(
        title=f"Samples per Second (avg over last {int(window_seconds)}s)",
        xaxis_title="Device",
        yaxis_title="Samples per second",
        hovermode='x unified',
        showlegend=False,
        height=520,
        width=520
    )
    return fig


def create_latency_figure(window_seconds):
    """Show average inference latency (ms) comparison: CPU vs FPGA."""
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    try:
        window_seconds = float(window_seconds)
    except (TypeError, ValueError):
        window_seconds = THROUGHPUT_WINDOW_SECONDS
    if window_seconds <= 0:
        window_seconds = THROUGHPUT_WINDOW_SECONDS

    latest_ts = realtime_data['timestamp'][-1]
    cutoff = latest_ts - timedelta(seconds=window_seconds)

    cpu_latencies = []
    alveo_latencies = []

    # Iterate backwards or zip through lists
    # Assuming lists are synchronized by index
    for i, ts in enumerate(realtime_data['timestamp']):
        if ts < cutoff:
            continue
        
        # Host latency
        if realtime_data['latency_ms'][i] is not None:
            cpu_latencies.append(realtime_data['latency_ms'][i])
        
        # Alveo latency
        if realtime_data['latency_ms_alveo'][i] is not None:
            alveo_latencies.append(realtime_data['latency_ms_alveo'][i])

    if not cpu_latencies and not alveo_latencies:
        return create_empty_figure("No Data Available")

    averages = {}
    if cpu_latencies:
        averages['CPU'] = sum(cpu_latencies) / len(cpu_latencies)
    if alveo_latencies:
        averages['FPGA'] = sum(alveo_latencies) / len(alveo_latencies)

    fig = go.Figure()
    labels = list(averages.keys())
    values = list(averages.values())
    
    # Colors: Host = Blueish, Alveo = Orange/Reddish
    colors = ['#7480ff' if 'CPU' in lbl else '#ff8b73' for lbl in labels]

    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        marker=dict(color=colors),
        text=[f"{v:.2f} ms" for v in values],
        textposition='auto'
    ))

    fig.update_layout(
        title=f"Avg Inference Latency (last {int(window_seconds)}s)",
        xaxis_title="Device",
        yaxis_title="Latency (ms)",
        hovermode='x unified',
        showlegend=False,
        height=520,
        width=520
    )
    return fig


def create_fpr_figure():
    """Plot running false positive rate over time."""
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    idx_sorted = sorted(range(len(realtime_data['timestamp'])), key=lambda i: realtime_data['timestamp'][i])
    timestamps = [realtime_data['timestamp'][i] for i in idx_sorted]
    anomalies_sorted = [realtime_data['is_anomaly'][i] for i in idx_sorted]
    labels_sorted = [realtime_data['true_label'][i] for i in idx_sorted]

    fp_count = 0
    tn_count = 0
    fpr_values = []
    for predicted, label in zip(anomalies_sorted, labels_sorted):
        if label == 0 and predicted:
            fp_count += 1
        if label == 0 and not predicted:
            tn_count += 1
        denominator = fp_count + tn_count
        fpr_values.append((fp_count / denominator) * 100 if denominator else 0.0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=fpr_values,
        mode='lines+markers',
        name='False Positive Rate',
        line=dict(color='purple'),
        marker=dict(size=6, color='purple')
    ))
    if fpr_values:
        fig.add_hline(
            y=fpr_values[-1],
            line_dash="dot",
            line_color="gray",
            annotation_text=f"Current FPR: {fpr_values[-1]:.1f}%",
            annotation_position="top right"
        )
    fig.update_layout(
        title="False Positive Rate Over Time",
        xaxis_title="Time",
        yaxis_title="FPR (%)",
        hovermode='x unified'
    )
    return fig


def create_feature_figure():
    """Build real-time network feature visualization comparing raw vs anonymized."""
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    idx_sorted = sorted(range(len(realtime_data['timestamp'])), key=lambda i: realtime_data['timestamp'][i])
    timestamps = [realtime_data['timestamp'][i] for i in idx_sorted]
    is_anomaly_sorted = [realtime_data['is_anomaly'][i] for i in idx_sorted]

    def _safe_series(series, indices):
        result = []
        for i in indices:
            if i < len(series):
                result.append(series[i])
        return result

    fig = go.Figure()

    # Limit visualization to downlink bitrate only
    feature_names = [f for f in ('dl_bitrate',) if f in realtime_data['feature_values']]

    base_colors = ['#0d6efd', '#20c997', '#fd7e14']

    for i, feature in enumerate(feature_names):
        display_name = FEATURE_DISPLAY_NAMES.get(feature, feature.replace('_', ' ').title())
        feature_color = FEATURE_COLORS.get(feature, base_colors[i % len(base_colors)])

        # ANONYMIZED (drawn first so raw overlay is legible)
        anon_series = realtime_data['feature_values'].get(feature, [])
        if anon_series:
            anon_sorted = _safe_series(anon_series, idx_sorted)
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=anon_sorted,
                mode='lines',
                name=f"{display_name} (anonymized)",
                line=dict(color=feature_color, width=3)
            ))

        # RAW
        raw_series = realtime_data['raw_feature_values'].get(feature, [])
        if raw_series:
            raw_sorted = _safe_series(raw_series, idx_sorted)
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=raw_sorted,
                mode='lines+markers',
                name=f"{display_name} (raw)",
                line=dict(color=RAW_TRACE_COLOR, dash='dash', width=2),
                marker=dict(symbol='circle-open', size=6, line=dict(width=1.5, color=RAW_TRACE_COLOR)),
                opacity=0.85
            ))

    # Highlight anomalies using the first feature’s anonymized values (as before)
    anomaly_times = [timestamps[i] for i, anomaly in enumerate(is_anomaly_sorted) if anomaly]
    if anomaly_times and feature_names:
        first_feature = feature_names[0]
        anon_series = _safe_series(realtime_data['feature_values'].get(first_feature, []), idx_sorted)
        if anon_series:
            anomaly_values = [anon_series[i]
                              for i, anomaly in enumerate(is_anomaly_sorted)
                              if anomaly and i < len(anon_series)]
            if anomaly_values:
                fig.add_trace(go.Scatter(
                    x=anomaly_times[:len(anomaly_values)],
                    y=anomaly_values,
                    mode='markers',
                    name='Detected Anomalies',
                    marker=dict(color='red', size=10, symbol='x'),
                    showlegend=True
                ))

    primary_label = FEATURE_DISPLAY_NAMES.get(feature_names[0], "Network Feature") if feature_names else "Network Feature"
    fig.update_layout(
        title=f"{primary_label} — Raw vs Anonymized",
        xaxis_title="Time",
        yaxis_title=f"{primary_label} Value",
        hovermode='x unified',
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.12,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(255,255,255,0.8)"
        ),
        margin=dict(t=90)
    )
    return fig


def create_anomaly_figure(source='cpu'):
    """
    Create reconstruction error plot.
    source: 'cpu' or 'alveo'
    """
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    idx_sorted = sorted(range(len(realtime_data['timestamp'])), key=lambda i: realtime_data['timestamp'][i])
    timestamps = [realtime_data['timestamp'][i] for i in idx_sorted]

    # Select data based on source
    if source == 'alveo':
        # Check if Alveo data exists (not None)
        valid_data = [x for x in realtime_data['score_alveo'] if x is not None]
        if not valid_data:
            return create_empty_figure("FPGA is not initialized")
            
        errors = [realtime_data['score_alveo'][i] for i in idx_sorted]
        is_anomaly_flags = [realtime_data['is_anomaly_alveo'][i] for i in idx_sorted]
        # Use alveo specific labels if available, else fallback to common
        true_labels = [realtime_data['true_label_alveo'][i] if realtime_data['true_label_alveo'][i] is not None 
                       else realtime_data['true_label'][i] for i in idx_sorted]
        title_suffix = "(FPGA)"
    else:
        errors = [realtime_data['reconstruction_error'][i] for i in idx_sorted]
        is_anomaly_flags = [realtime_data['is_anomaly'][i] for i in idx_sorted]
        true_labels = [realtime_data['true_label'][i] for i in idx_sorted]
        title_suffix = "(CPU)"

    # Handle case where filtered data might be None in the list (if mixed)
    # Replace None with 0.0 or skip for plotting safety
    clean_errors = []
    clean_timestamps = []
    clean_flags = []
    clean_labels = []
    
    for t, e, f, l in zip(timestamps, errors, is_anomaly_flags, true_labels):
        if e is not None:
            clean_timestamps.append(t)
            clean_errors.append(e)
            clean_flags.append(f)
            clean_labels.append(l)

    if not clean_timestamps:
        return create_empty_figure(f"No {source.upper()} Data")

    fig = go.Figure()

    colors = ['red' if anomaly else 'blue' for anomaly in clean_flags]

    fig.add_trace(go.Scatter(
        x=clean_timestamps,
        y=clean_errors,
        mode='markers+lines',
        name='Reconstruction Error',
        marker=dict(color=colors, size=6),
        line=dict(color='gray', width=1)
    ))

    # Add threshold line
    fig.add_hline(
        y=detector.threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Threshold ({detector.threshold:.6f})"
    )

    # Add ground truth markers
    true_anomaly_times = [clean_timestamps[i] for i, label in enumerate(clean_labels) if label == 1]
    true_anomaly_scores = [clean_errors[i] for i, label in enumerate(clean_labels) if label == 1]

    if true_anomaly_times:
        fig.add_trace(go.Scatter(
            x=true_anomaly_times,
            y=true_anomaly_scores,
            mode='markers',
            name='True Attacks',
            marker=dict(color='orange', size=8, symbol='diamond'),
            showlegend=True
        ))

    fig.update_layout(
        title=f"Reconstruction Error {title_suffix}",
        xaxis_title="Time",
        yaxis_title="L1 Loss",
        hovermode='x unified',
        margin=dict(t=40, b=20)
    )

    return fig


def create_diff_figure():
    """Create a plot showing absolute difference between CPU and Alveo scores."""
    if not realtime_data['timestamp']:
        return create_empty_figure("No Data Available")

    idx_sorted = sorted(range(len(realtime_data['timestamp'])), key=lambda i: realtime_data['timestamp'][i])
    
    diffs = []
    timestamps = []
    
    has_valid_alveo = False

    for i in idx_sorted:
        cpu_score = realtime_data['reconstruction_error'][i]
        alveo_score = realtime_data['score_alveo'][i]
        
        if cpu_score is not None and alveo_score is not None:
            has_valid_alveo = True
            diffs.append(abs(cpu_score - alveo_score))
            timestamps.append(realtime_data['timestamp'][i])
    
    if not has_valid_alveo:
        return create_empty_figure("FPGA is not initialized")

    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=diffs,
        mode='markers+lines',
        name='Abs Diff',
        line=dict(color='#fd7e14', width=2),
        marker=dict(size=4)
    ))
    
    fig.update_layout(
        title="Reconstruction Error Difference (CPU vs FPGA)",
        xaxis_title="Time",
        yaxis_title="|CPU - FPGA|",
        hovermode='x unified',
        margin=dict(t=40, b=20)
    )
    
    return fig


def create_statistics():
    """Calculate and display detection performance metrics."""
    if not realtime_data['timestamp']:
        return html.P("No data available")

    total_points = len(realtime_data['timestamp'])
    detected_anomalies = sum(realtime_data['is_anomaly'])
    true_attacks = sum(1 for label in realtime_data['true_label'] if label == 1)

    # True positive rate
    true_positives = sum(1 for i in range(len(realtime_data['is_anomaly']))
                         if realtime_data['is_anomaly'][i] and realtime_data['true_label'][i] == 1)

    true_positive_rate = (true_positives / true_attacks) * 100 if true_attacks > 0 else 0

    # False positive rate
    false_positives = sum(1 for i in range(len(realtime_data['is_anomaly']))
                          if realtime_data['is_anomaly'][i] and realtime_data['true_label'][i] == 0)

    normal_samples = total_points - true_attacks
    false_positive_rate = (false_positives / normal_samples) * 100 if normal_samples > 0 else 0

    return dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("📊 Total Samples"),
                    html.H3(f"{total_points}", className="text-primary")
                ])
            ])
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("🚨 Detected"),
                    html.H3(f"{detected_anomalies}", className="text-danger")
                ])
            ])
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("✅ TPR"),
                    html.H3(f"{true_positive_rate:.1f}%", className="text-success")
                ])
            ])
        ], width=3),

        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    html.H5("❌ FPR"),
                    html.H3(f"{false_positive_rate:.1f}%", className="text-info")
                ])
            ])
        ], width=3)
    ])


def create_device_list():
    """Show anonymized device list with anomaly rates."""
    if not realtime_data['anonymized_device_id']:
        return html.P("No devices detected yet", className="text-muted")

    # Get unique anonymized device IDs and their anomaly counts
    device_counts = {}
    for i, device_id in enumerate(realtime_data['anonymized_device_id']):
        if device_id not in device_counts:
            device_counts[device_id] = {'total': 0, 'anomalies': 0}
        device_counts[device_id]['total'] += 1
        if realtime_data['is_anomaly'][i]:
            device_counts[device_id]['anomalies'] += 1

    # Create list items
    device_items = []
    for device_id, counts in sorted(device_counts.items())[-10:]:  # Show last 10 devices
        anomaly_rate = (counts['anomalies'] / counts['total']) * 100 if counts['total'] > 0 else 0
        color = "danger" if anomaly_rate > 50 else "warning" if anomaly_rate > 20 else "success"

        device_items.append(
            dbc.ListGroupItem([
                html.Div([
                    html.Span(f"Device: {device_id}", className="fw-bold"),
                    dbc.Badge(f"{anomaly_rate:.0f}%", color=color, className="float-end")
                ]),
                html.Small(f"Samples: {counts['total']}, Anomalies: {counts['anomalies']}",
                           className="text-muted")
            ])
        )

    return dbc.ListGroup(device_items, flush=True)


if __name__ == '__main__':
    logging.info("🛡️ PRIVATEER Federated Learning Model Integrated Dashboard")
    logging.info("=" * 50)
    logging.info("🤖 Using TransformerAD Model with Differential Privacy")
    logging.info(f"📱 Device: {detector.device}")
    logging.info(f"🎯 Initial Threshold: {detector.threshold}")
    logging.info(f"📊 Input Features: {detector.input_features}")
    logging.info(f"📄 Dataset: Loaded via DataProcessor.get_dataloader('test')")
    logging.info("🔐 Privacy Protection: Anonymization Active")
    logging.info("=" * 50)
    logging.info("Starting web server...")
    logging.info("Open your browser and go to: http://127.0.0.1:8056")
    logging.info("=" * 50)

    app.run(host='127.0.0.1', port=8056, debug=True)