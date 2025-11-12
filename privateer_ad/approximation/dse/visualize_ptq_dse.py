# privateer_ad/approximation/dse/visualize_ptq_dse.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient

import matplotlib.pyplot as plt
import seaborn as sns

from privateer_ad.config import PathConfig, MLFlowConfig


DEFAULT_TARGET_METRICS = [
    "approx_f1-score",
    "anomaly_score_abs_diff_mean",
    "approx_roc_auc",
    "approx_attack_success_rate_eps_0.01",
]


def _ensure_output_dir(experiment_name: str, output_root: Optional[Path | str] = None) -> Path:
    pc = PathConfig()
    base = Path(output_root) if output_root else pc.experiments_dir
    out = base / "dse_visualizations" / experiment_name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _fmt_for_metric(metric: str) -> str:
    """Choose a sensible annotation format per metric."""
    if "diff" in metric:
        return ".6f"
    return ".3f"


def _sanitize(s: str) -> str:
    return s.replace("/", "_").replace("\\", "_").replace(" ", "_").replace(":", "_")


def _collect_runs_df(
    client: MlflowClient,
    experiment_name: str,
    target_metrics: List[str],
) -> pd.DataFrame:
    exp = client.get_experiment_by_name(experiment_name)
    if exp is None:
        raise ValueError(f"MLflow experiment '{experiment_name}' not found.")

    runs = client.search_runs([exp.experiment_id], max_results=50000)

    rows = []
    for r in runs:
        params = r.data.params
        metrics = r.data.metrics

        # Required params
        if "weight_bits" not in params or "activation_bits" not in params:
            continue

        row = {
            "run_id": r.info.run_id,
            "run_name": r.info.run_name or "",
            "weight_bits": int(params["weight_bits"]),
            "activation_bits": int(params["activation_bits"]),
        }

        has_any_metric = False
        for m in target_metrics:
            if m in metrics:
                row[m] = float(metrics[m])
                has_any_metric = True
            else:
                row[m] = np.nan  # keep column shape; may be NaN

        if has_any_metric:
            rows.append(row)

    if not rows:
        raise RuntimeError(
            f"No runs with required params/metrics found in experiment '{experiment_name}'."
        )

    df = pd.DataFrame(rows)
    df["total_bits"] = df["weight_bits"] + df["activation_bits"]
    return df


def _plot_heatmap(
    df: pd.DataFrame,
    metric: str,
    experiment_name: str,
    out_dir: Path,
    cmap: str = "viridis",
    annotate: bool = True,
) -> Path:
    pivot = (
        df.pivot_table(
            index="activation_bits",
            columns="weight_bits",
            values=metric,
            aggfunc="mean",
        )
        .sort_index(ascending=True)
        .sort_index(axis=1, ascending=True)
    )

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        pivot,
        cmap=cmap,
        annot=annotate,
        fmt=_fmt_for_metric(metric),
        cbar=True,
        linewidths=0.5,
        linecolor="white",
    )
    plt.title(f"{metric} — {experiment_name}\n(y = Activation bits, x = Weight bits)")
    plt.ylabel("Activation bits")
    plt.xlabel("Weight bits")
    plt.tight_layout()

    fout = out_dir / f"heatmap_{_sanitize(metric)}.png"
    plt.savefig(fout, dpi=200)
    plt.close()
    return fout


def _plot_total_bits_curve(
    df: pd.DataFrame,
    metric: str,
    experiment_name: str,
    out_dir: Path,
) -> Path:
    # Aggregate by total_bits
    grp = (
        df.groupby("total_bits")[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
        .sort_values("total_bits")
    )

    # Plot: raw scatter + mean line + std band
    plt.figure(figsize=(9, 5))
    # raw points (slight jitter to reduce overlap)
    jitter = (np.random.rand(len(df)) - 0.5) * 0.2
    plt.scatter(df["total_bits"] + jitter, df[metric], alpha=0.35)

    # mean line + std fill
    plt.plot(grp["total_bits"], grp["mean"], marker="o")
    if (grp["std"].notna()).any():
        y1 = grp["mean"] - grp["std"]
        y2 = grp["mean"] + grp["std"]
        plt.fill_between(grp["total_bits"], y1, y2, alpha=0.2)

    plt.title(f"{metric} vs. total bits (W+A) — {experiment_name}")
    plt.xlabel("Total bits = weight_bits + activation_bits")
    plt.ylabel(metric)
    plt.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    fout = out_dir / f"total_bits_{_sanitize(metric)}.png"
    plt.savefig(fout, dpi=200)
    plt.close()
    return fout


def visualize_ptq_dse(
    experiment_name: str,
    tracking_uri: Optional[str] = None,
    target_metrics: Optional[List[str]] = None,
    output_root: Optional[Path | str] = None,
    cmap: str = "viridis",
    annotate_heatmaps: bool = True,
) -> Dict[str, List[str]]:
    """
    Build PTQ DSE visualizations from an MLflow experiment.

    - Heatmaps for each metric (x=weight_bits, y=activation_bits).
    - Curves for each metric vs total (weight_bits + activation_bits).

    Saves all artifacts under:
        experiments/dse_visualizations/{experiment_name}

    Returns:
        dict mapping metric -> list of saved file paths (strings).
    """
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    metrics = target_metrics or list(DEFAULT_TARGET_METRICS)
    out_dir = _ensure_output_dir(experiment_name, output_root)

    # Collect runs
    df = _collect_runs_df(client, experiment_name, metrics)

    # Persist the compiled runs table for traceability
    compiled_csv = out_dir / "compiled_runs.csv"
    df.to_csv(compiled_csv, index=False)

    # Make plots
    saved: Dict[str, List[str]] = {}
    for m in metrics:
        files = []
        try:
            files.append(str(_plot_heatmap(df, m, experiment_name, out_dir, cmap, annotate_heatmaps)))
            files.append(str(_plot_total_bits_curve(df, m, experiment_name, out_dir)))
        except Exception as e:
            print(f"[WARN] Skipped plotting for '{m}': {e}")
        saved[m] = files

    # Minimal metadata
    meta = {
        "experiment_name": experiment_name,
        "n_runs": int(df.shape[0]),
        "metrics": metrics,
        "output_dir": str(out_dir),
        "compiled_csv": str(compiled_csv),
    }
    pd.Series(meta).to_json(out_dir / "meta.json", indent=2)

    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize PTQ DSE results from MLflow.")
    parser.add_argument("experiment_name", help="Name of the MLflow experiment to visualize.")
    parser.add_argument("--tracking-uri", default=MLFlowConfig().tracking_uri, help="MLflow tracking URI (optional).")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=DEFAULT_TARGET_METRICS,
        help="Metrics to visualize (default: %(default)s)",
    )
    parser.add_argument("--output-root", default=None, help="Override output root dir (optional).")
    parser.add_argument("--no-annotate", action="store_true", help="Disable heatmap annotations.")
    parser.add_argument("--cmap", default="viridis", help="Matplotlib colormap for heatmaps.")
    args = parser.parse_args()

    visualize_ptq_dse(
        experiment_name=args.experiment_name,
        tracking_uri=args.tracking_uri,
        target_metrics=args.metrics,
        output_root=args.output_root,
        cmap=args.cmap,
        annotate_heatmaps=not args.no_annotate,
    )