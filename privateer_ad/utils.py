import logging
from pathlib import Path

import mlflow
import numpy as np
import torch

from privateer_ad.config import TrainingConfig


_THRESHOLD_METRIC_KEYS = [
    "global_test_threshold",
    "val_threshold",
    "test_threshold",
    "threshold",
]
_DEFAULT_THRESHOLD = 0.5


def load_model_weights(model_path: str, paths_config) -> dict:
    """
    Load PyTorch model weights with flexible path resolution and format handling.

    Attempts to locate and load model weights from various common path patterns,
    handling different state dictionary formats and cleaning distributed training
    artifacts for compatibility with single-device loading.

    Args:
        model_path (str): Model file path or experiment identifier
        paths_config: Path configuration object with directory specifications

    Returns:
        dict: Cleaned model state dictionary ready for loading

    Raises:
        FileNotFoundError: When model file cannot be located in any expected location
    """
    model_path = Path(model_path)

    # Try different path resolutions
    possible_paths = [
        model_path,  # Direct path
        paths_config.experiments_dir / model_path / 'model.pt',  # Experiment directory
        paths_config.models_dir / model_path,  # Models directory
        Path(model_path).with_suffix('.pt'),  # Add .pt extension
    ]

    attempted_paths: list[str] = []

    for path in possible_paths:
        if not path.exists():
            attempted_paths.append(str(path))
            continue

        candidate = path
        if path.is_dir():
            candidate = path / 'model.pt'
            attempted_paths.append(str(candidate))
            if not candidate.exists():
                logging.debug(f'Directory {path} does not contain model.pt; skipping.')
                continue
        else:
            attempted_paths.append(str(candidate))

        if not candidate.is_file():
            logging.debug(f'Path {candidate} is not a file; skipping.')
            continue

        logging.info(f'Loading model from: {candidate}')
        try:
            checkpoint = torch.load(candidate, map_location='cpu', weights_only=False)

            # Support checkpoints that wrap the state dict or store nn.Module
            if isinstance(checkpoint, torch.nn.Module):
                raw_state = checkpoint.state_dict()
            elif isinstance(checkpoint, dict):
                if 'state_dict' in checkpoint and isinstance(checkpoint['state_dict'], dict):
                    raw_state = checkpoint['state_dict']
                elif 'model_state_dict' in checkpoint and isinstance(checkpoint['model_state_dict'], dict):
                    raw_state = checkpoint['model_state_dict']
                else:
                    raw_state = checkpoint
            else:
                raise ValueError(f'Unexpected checkpoint format: {type(checkpoint)}')

            cleaned_state_dict = {}
            for key, value in raw_state.items():
                clean_key = key
                for prefix in ['_module.', 'module.']:
                    if clean_key.startswith(prefix):
                        clean_key = clean_key[len(prefix):]
                cleaned_state_dict[clean_key] = value

            return cleaned_state_dict

        except Exception as e:
            logging.warning(f'Failed to load from {candidate}: {e}')
            continue

    raise FileNotFoundError(f'Could not find model file at any of: {attempted_paths}')


def log_model(model, model_name, sample, direction, target_metric, current_metrics, experiment_id, pip_requirements):
    """
    Log trained model to MLflow with champion tracking and version management.

    Registers the model in MLflow with champion tagging based on
    performance comparison with previous versions. Manages version lifecycle
    and metadata tagging for model selection and deployment decisions.

    Args:
        model: Trained PyTorch model to register
        model_name (str): Registry name for the model
        sample: Representative input sample for signature inference
        direction (str): Optimization direction ('maximize' or 'minimize')
        target_metric (str): Primary metric for champion selection
        current_metrics (dict): Performance metrics from current training
        experiment_id (str): MLflow experiment identifier
        pip_requirements (str): Path to requirements file for reproducibility

    Note:
        Automatically manages champion tag transfer between model versions
        based on performance improvement detection.
    """
    model.to('cpu')
    signature = get_signature(model=model, sample=sample)
    client = mlflow.tracking.MlflowClient()

    # Determine sort direction for finding best run
    # Be robust: if the requested target_metric is not present (e.g., server has only
    # global_test_* metrics after aggregation), fall back to a sensible available metric.
    selected_metric_key = target_metric
    if target_metric not in current_metrics:
        # Try common alternates in order of preference
        candidates = []
        try:
            if target_metric.startswith('val_'):
                suffix = target_metric[len('val_'):]
                candidates.extend([
                    f'global_test_{suffix}',
                    f'test_{suffix}',
                    suffix,
                ])
        except Exception:
            pass
        # Generic fallbacks commonly produced by server/client eval
        candidates.extend([
            'global_test_f1-score',
            'test_f1-score',
            'val_f1-score',
            'f1-score',
        ])

        for cand in candidates:
            if cand in current_metrics:
                selected_metric_key = cand
                logging.warning(
                    f"Requested target_metric '{target_metric}' not found in current_metrics. "
                    f"Falling back to '{selected_metric_key}'."
                )
                break
        else:
            # As a last resort, pick any numeric metric
            for k, v in current_metrics.items():
                if isinstance(v, (int, float)):
                    selected_metric_key = k
                    logging.warning(
                        f"Requested target_metric '{target_metric}' not found. "
                        f"Falling back to first numeric metric '{selected_metric_key}'."
                    )
                    break

    current_target_metric = current_metrics[selected_metric_key]
    sorting = 'DESC' if direction == 'maximize' else 'ASC'

    # Check if this is a new champion
    is_champion = True
    best_target_metric = -np.inf if direction == 'maximize' else np.inf
    champions = []

    try:
        # Find the best run across all experiments
        best_run = client.search_runs([experiment_id],
                                      order_by=[f'metrics.`{target_metric}` {sorting}'],
                                      max_results=1)[0]

        if target_metric in best_run.data.metrics:
            best_target_metric = best_run.data.metrics[target_metric]

            # Compare with previous best
            if direction == 'maximize':
                is_champion = current_target_metric >= best_target_metric
            else:
                is_champion = current_target_metric <= best_target_metric

            logging.info(f'Previous best {selected_metric_key}: {best_target_metric}')
            logging.info(f'Current {selected_metric_key}: {current_target_metric}')
            logging.info(f'Is new champion: {is_champion}')

        # Find previous champion versions to remove tag
        try:
            champions = client.search_model_versions(filter_string=f"name = '{model_name}' and tag.champion = 'true'")
        except Exception as e:
            logging.warning(f"No previous champion found or error accessing registry: {e}")

    except (IndexError, KeyError) as e:
        logging.warning(f"No previous runs found or error accessing metrics: {e}")
        # This is likely the first model, so it's automatically the champion
        is_champion = True

    # Log the model
    model_info = mlflow.pytorch.log_model(
        pytorch_model=model,
        artifact_path=model_name,
        registered_model_name=model_name,
        signature=signature,
        pip_requirements=pip_requirements
    )

    current_version = model_info.registered_model_version

    # Handle champion tagging
    if is_champion:
        logging.info(f"🏆 New champion model! Logging with champion tag.")

        # Remove champion tag from previous versions
        for champion in champions:
            try:
                client.delete_model_version_tag(model_name, champion.version, 'champion')
                logging.info(f"Removed champion tag from version {champion.version}")
            except Exception as e:
                logging.warning(f"Failed to remove champion tag from version {champion.version}: {e}")

        try:
            client.set_model_version_tag(model_name, current_version, 'champion', 'true')

            # Add metadata about when it became champion
            import datetime
            client.set_model_version_tag(
                model_name,
                current_version,
                'champion_since',
                datetime.datetime.now().isoformat()
            )

            logging.info(f"Added champion tag to version {current_version}")
        except Exception as e:
            logging.error(f"Error adding champion tag: {e}")
    else:
        logging.info(f"Model performance ({current_target_metric}) did not exceed "
                     f"previous best ({best_target_metric}). No champion tag added.")

    # Record which metric was actually used for champion decision
    client.set_model_version_tag(model_name, current_version, selected_metric_key, str(current_target_metric))


def get_signature(model, sample):
    """Generate MLflow model signature from sample input and model output."""
    _output = model(sample)

    if isinstance(_output, dict):
        _output = {key: val.detach().numpy() for key, val in _output.items()}
    else:
        _output = _output.detach().numpy()
    signature = mlflow.models.infer_signature(model_input=sample.detach().numpy(), model_output=_output)
    return signature


def _extract_threshold_from_run(run):
    """Return (threshold_value, metric_key) tuple from a run's recorded metrics."""
    for key in _THRESHOLD_METRIC_KEYS:
        value = run.data.metrics.get(key)
        if value is not None:
            return float(value), key
    return None, None


def _build_loss_fn_from_run(run):
    """Instantiate the reconstruction loss configured for the given run."""
    loss_fn_name = run.data.params.get("loss_fn_name", TrainingConfig().loss_fn_name)
    loss_fn = getattr(torch.nn, loss_fn_name)(reduction="none")
    return loss_fn, loss_fn_name


def _resolve_experiment_ids(client, experiment_name):
    """Determine which experiment IDs to search when looking up runs."""
    if experiment_name:
        experiment = client.get_experiment_by_name(experiment_name)
        if experiment is not None:
            return [experiment.experiment_id]
        logging.warning("Experiment '%s' not found; searching all available experiments.", experiment_name)

    experiments = client.search_experiments()
    return [exp.experiment_id for exp in experiments if exp.lifecycle_stage != "deleted"]


def _find_run_by_name(client, run_name, experiment_name):
    """Locate the newest run with the requested friendly name."""
    experiment_ids = _resolve_experiment_ids(client, experiment_name)
    filter_string = f"tags.mlflow.runName = '{run_name}'"
    runs = client.search_runs(
        experiment_ids=experiment_ids,
        filter_string=filter_string,
        order_by=["attribute.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise ValueError(f"No MLflow run found with name '{run_name}'.")
    return runs[0]


def load_mlflow_model_from_run(
    tracking_uri,
    *,
    run_id=None,
    run_name=None,
    artifact_path="global_TransformerAD",
    experiment_name=None,
):
    """
    Load a specific MLflow run's model artifact.

    Args:
        tracking_uri (str): Backend/tracking store URI.
        run_id (str, optional): Explicit MLflow run ID to load.
        run_name (str, optional): Friendly MLflow run name (e.g. 'awesome-roo-953').
        artifact_path (str): Model artifact path relative to the run (default: global_TransformerAD).
        experiment_name (str, optional): Restrict search to this experiment when resolving by run name.

    Returns:
        tuple(model, threshold, loss_fn, run): Loaded PyTorch module, stored threshold,
        configured reconstruction loss, and the MLflow Run object.
    """
    if not run_id and not run_name:
        raise ValueError("Either run_id or run_name must be provided to load a run.")

    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    if run_id:
        run = client.get_run(run_id)
    else:
        run = _find_run_by_name(client, run_name, experiment_name)
        run_id = run.info.run_id

    map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_uri = f"runs:/{run_id}/{artifact_path}"
    model = mlflow.pytorch.load_model(model_uri=model_uri, map_location=map_location)

    threshold, threshold_key = _extract_threshold_from_run(run)
    if threshold is None:
        logging.warning("No threshold metric logged for run %s; defaulting to %.2f.", run_id, _DEFAULT_THRESHOLD)
        threshold = _DEFAULT_THRESHOLD

    loss_fn, loss_fn_name = _build_loss_fn_from_run(run)

    logging.info(
        "Loaded MLflow run %s (name=%s) artifact=%s threshold=%.6f [%s] loss_fn=%s",
        run_id,
        run.data.tags.get("mlflow.runName"),
        artifact_path,
        threshold,
        threshold_key or "default",
        loss_fn_name,
    )

    return model, threshold, loss_fn, run


def load_champion_model(tracking_uri, model_name: str = "TransformerAD"):
    """
    Load the best-performing model version with associated metadata.

    Retrieves the champion model from MLflow registry along with its optimal
    threshold and loss function configuration. Falls back to latest version
    if no champion is tagged.

    Args:
        tracking_uri (str): MLflow tracking server address
        model_name (str): Registered model name to load

    Returns:
        tuple: (model, threshold, loss_fn) for complete inference setup

    Raises:
        Exception: When model loading or metadata extraction fails
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    try:
        # Get registered model
        version = None
        run_id = None
        source = None
        # Get all versions of the model
        all_versions = client.search_model_versions(f"name='{model_name}'")
        logging.info(f'Found {len(all_versions)} total versions for model {model_name}')

        # Find champion version
        for model_version in all_versions:
            if model_version.tags.get('champion') == 'true':
                version = model_version.version
                run_id = model_version.run_id
                source = model_version.source
                logging.info(f"🏆 Found champion model v{version}")
                break

        if version is None:
            # Sort by version number and get latest
            latest_version = max(all_versions, key=lambda v: int(v.version))
            version = latest_version.version
            run_id = latest_version.run_id
            source = latest_version.source
            logging.warning(f"⚠️ No champion found, using latest version v{version}")

        # Load the model
        logging.info(f"Loading model from: {source}")
        load_conf = {'map_location': torch.device('cuda' if torch.cuda.is_available() else 'cpu')}
        model = mlflow.pytorch.load_model(model_uri=source, **load_conf)
        logging.info(f"Model loaded!")

        # Get run details to extract threshold and loss_fn
        run = client.get_run(run_id)
        threshold, threshold_key = _extract_threshold_from_run(run)
        if threshold is None:
            logging.warning("⚠️ No threshold found in run metrics, using default: %.2f", _DEFAULT_THRESHOLD)
            threshold = _DEFAULT_THRESHOLD
        else:
            logging.info("📏 Found threshold: %.6f (from metric: %s)", threshold, threshold_key)

        loss_fn, loss_fn_name = _build_loss_fn_from_run(run)

        logging.info(f"✅ Successfully loaded:")
        logging.info(f"🏆 Model: {model_name} v{version}")
        logging.info(f"📏 Threshold: {threshold}")
        logging.info(f"📊 Loss function: {loss_fn_name}")

        return model, threshold, loss_fn

    except Exception as e:
        logging.error(f"❌ Error loading champion model: {e}")
        raise