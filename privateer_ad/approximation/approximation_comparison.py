# privateer_ad/evaluate/approximation_comparison.py
import logging
from typing import Any, Dict, List, Optional, Union, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import mlflow
from sklearn.metrics import classification_report

from privateer_ad.alveo.alveo_runner import AlveoRunner
from privateer_ad.evaluate.evaluator import ModelEvaluator
from privateer_ad.evaluate.evaluator_alveo import AlveoEvaluator
from privateer_ad.approximation.transformer_ad_fxp import TransformerADQConfig
from privateer_ad.robustness.evaluator import evaluate_robustness
from privateer_ad.config import ModelConfig


def _fig_line(values: List[float], title: str, ylabel: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(values)
    ax.set_title(title)
    ax.set_xlabel("sample index")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig


def _fig_hist(values: List[float], title: str, xlabel: str, bins: int = 60) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(values, bins=bins, alpha=0.85)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("count")
    ax.grid(True, alpha=0.3, linestyle="--")
    fig.tight_layout()
    return fig


def approximation_comparison(
    golden_model: torch.nn.Module,
    approximated: Union[torch.nn.Module, AlveoRunner],
    dataloader,
    threshold: float,
    device: Optional[Union[str, torch.device]] = None,
    loss_fn_name: str = "L1Loss",
    calculated_golden: Optional[Tuple[Dict, Dict, np.ndarray]] = None,
    return_golden_scores: bool = False,
) -> Union[Dict[str, Any], Tuple[Dict[str, Any], np.ndarray]]:
    """
    Compare a 'golden' PyTorch model vs an approximated model (PyTorch or AlveoRunner).

    This function reuses the standard evaluation pipelines to avoid recomputing
    model outputs, focusing only on the difference in their final anomaly scores.

    Steps:
      1) Evaluate both models with their respective evaluators, capturing the
         per-sample anomaly scores.
      2) Compute the absolute difference between the anomaly score arrays.
      3) Aggregate the differences (mean, std).
      4) Produce figures visualizing the per-sample score differences
         (line plot) and their distribution (histogram).

    Args:
        # ... (other args)
        calculated_golden (Optional[Tuple]): Pre-computed results for the golden model
                                             in the format (metrics, figures, scores)
                                             to avoid re-calculation.
        return_golden_scores (bool): If True, returns the golden_scores as a second
                                     element in a tuple.

    Returns:
      A dictionary with the results, or a tuple (results_dict, golden_scores)
      if return_golden_scores is True.
      results_dict is:
        - 'golden': {'metrics': ..., 'figures': ...} from the evaluation run.
        - 'approx': {'metrics': ..., 'figures': ...} from the evaluation run.
        - 'comparison': {
              'per_sample': {
                  'anomaly_score_abs_diff': List[float],
              },
              'summary': {
                  'anomaly_score_abs_diff_mean': float,
                  'anomaly_score_abs_diff_std': float,
              },
              'figures': {
                  'anomaly_diff_line': Figure,
                  'anomaly_diff_hist': Figure,
              }
          }
    """
    # -----------------------
    # Setup
    # -----------------------
    device = torch.device(device) if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # -----------------------
    # 1) Full evaluator runs
    # -----------------------
    # Golden: use pre-calculated results if provided
    if calculated_golden:
        logging.info("Using pre-calculated results for the golden model.")
        golden_metrics, golden_figures, golden_scores = calculated_golden
    else:
        logging.info("Calculating results for the golden model.")
        golden_model = golden_model.to(device)
        golden_eval = ModelEvaluator(device=device, loss_fn=loss_fn_name)
        golden_metrics, golden_figures, golden_scores = golden_eval.evaluate(
            model=golden_model, dataloader=dataloader, threshold=threshold, prefix="golden", step=0, return_anomaly_scores=True
        )

    # Approx: either torch model or Alveo
    approx_metrics, approx_figures = {}, {}
    is_alveo = isinstance(approximated, AlveoRunner)

    if is_alveo:
        approx_eval = AlveoEvaluator(loss_fn=loss_fn_name)
        approx_metrics, approx_figures, approx_scores = approx_eval.evaluate(
            runner=approximated, dataloader=dataloader, threshold=threshold, prefix="approx", step=0, return_anomaly_scores=True
        )
    else:
        approximated = approximated.to(device)
        approx_eval = ModelEvaluator(device=device, loss_fn=loss_fn_name)
        approx_metrics, approx_figures, approx_scores = approx_eval.evaluate(
            model=approximated, dataloader=dataloader, threshold=threshold, prefix="approx", step=0, return_anomaly_scores=True
        )

    # -------------------------------------------------------
    # 2) Per-sample comparison of anomaly scores
    # -------------------------------------------------------
    anomaly_abs_diffs = np.abs(golden_scores - approx_scores).tolist()

    # ----------------------------------------
    # 3) Aggregate: mean & std of the series
    # ----------------------------------------
    diffs_arr = np.asarray(anomaly_abs_diffs, dtype=np.float64)

    comparison_summary = {
        "anomaly_score_abs_diff_mean": float(diffs_arr.mean())
        if diffs_arr.size
        else float("nan"),
        "anomaly_score_abs_diff_std": float(diffs_arr.std(ddof=1))
        if diffs_arr.size > 1
        else float("nan"),
    }

    # ----------------
    # 4) Make figures
    # ----------------
    figs = {
        "anomaly_diff_line": _fig_line(
            anomaly_abs_diffs, "Per-sample |anomaly score| difference", "|score_golden - score_approx|"
        ),
        "anomaly_diff_hist": _fig_hist(
            anomaly_abs_diffs, "Distribution of |anomaly score| differences", "|score_golden - score_approx|"
        ),
    }

    # -------------
    # Final bundle
    # -------------
    result = {
        "golden": {"metrics": golden_metrics, "figures": golden_figures},
        "approx": {"metrics": approx_metrics, "figures": approx_figures},
        "comparison": {
            "per_sample": {
                "anomaly_score_abs_diff": anomaly_abs_diffs,
            },
            "summary": comparison_summary,
            "figures": figs,
        },
    }
    logging.info(
        "Approximation comparison summary: "
        + ", ".join([f"{k}={v:.6f}" for k, v in comparison_summary.items() if isinstance(v, float)])
    )
    if return_golden_scores:
        return result, golden_scores
    return result

def approximation_comparison_mlflow(
    golden_model: torch.nn.Module,
    approximated: Union[torch.nn.Module, AlveoRunner],
    dataloader,
    threshold: float,
    mlflow_run_name: str,
    mlflow_params: dict,
    q_config: TransformerADQConfig = None,
    device: Optional[Union[str, torch.device]] = None,
    loss_fn_name: str = "L1Loss",
    check_robustness_approx: bool = False,
    check_robustness_golden: bool = False,
    epsilons: List[float] = [0.01],
    eps_step: float = 0.0005,
    max_iter: int = 100,
    calculated_golden: Optional[Tuple] = None,
    calculated_golden_robustness: Optional[Dict] = None,
) -> Dict[str, Any]:
    """
    Runs approximation_comparison and logs all parameters, metrics, and artifacts to MLflow.
    """
    try:
        mlflow.start_run(run_name=mlflow_run_name)
        logging.info(f"Started MLflow run: {mlflow_run_name}")

        # Log DSE parameters
        mlflow.log_params(mlflow_params)

        # Log the calibrated QConfig as an artifact
        if q_config:
            mlflow.log_dict(q_config.model_dump(mode="json"), "qconfig_calibrated.json")

        # Run the core comparison
        result = approximation_comparison(
            golden_model=golden_model,
            approximated=approximated,
            dataloader=dataloader,
            threshold=threshold,
            device=device,
            loss_fn_name=loss_fn_name,
            calculated_golden=calculated_golden
        )

        # Log all metrics
        mlflow.log_metrics(result["golden"]["metrics"])
        mlflow.log_metrics(result["approx"]["metrics"])
        mlflow.log_metrics(result["comparison"]["summary"])

        # Robustness Evaluation
        if check_robustness_golden:
            mlflow.log_param("robustness_epsilons", str(epsilons))
            mlflow.log_param("robustness_eps_step", eps_step)
            mlflow.log_param("robustness_max_iter", max_iter)
            # Golden Robustness Evaluation
            if calculated_golden_robustness:
                robustness_metrics_golden = calculated_golden_robustness
                logging.info("Using pre-calculated golden robustness metrics.")
            else:
                logging.info("Checking adversarial robustness for golden model...")
                robustness_metrics_golden = evaluate_robustness(
                    model=golden_model,
                    model_config=ModelConfig(),
                    dataloader=dataloader,
                    threshold=threshold,
                    epsilons=epsilons,
                    eps_step=eps_step,
                    max_iter=max_iter,
                    device=device,
                )
            mlflow.log_metrics({f"golden_{k}": v for k, v in robustness_metrics_golden.items()})
            result["golden_robustness_metrics"] = robustness_metrics_golden

            # Approximated Model Robustness Evaluation
            if not isinstance(approximated, AlveoRunner):
                logging.info("Checking adversarial robustness for approximated model...")
                robustness_metrics_approx = evaluate_robustness(
                    model=approximated,
                    model_config=ModelConfig(),
                    dataloader=dataloader,
                    threshold=threshold,
                    epsilons=epsilons,
                    eps_step=eps_step,
                    max_iter=max_iter,
                    device=device,
                )
                mlflow.log_metrics({f"approx_{k}": v for k, v in robustness_metrics_approx.items()})
                logging.info(f"Approx robustness metrics: {robustness_metrics_approx}")
                result["approx_robustness_metrics"] = robustness_metrics_approx

        # Log all figures.
        for name, fig in result["golden"]["figures"].items():
            # mlflow.log_figure(fig, f"golden_{name}.png")
            plt.close(fig)
        for name, fig in result["approx"]["figures"].items():
            # mlflow.log_figure(fig, f"approx_{name}.png")
            plt.close(fig)
        for name, fig in result["comparison"]["figures"].items():
            mlflow.log_figure(fig, f"comparison_{name}.png")
            plt.close(fig)
        
        logging.info("Successfully logged all results to MLflow.")
        return result

    except Exception as e:
        logging.error(f"MLflow logging or evaluation failed for run {mlflow_run_name}: {e}", exc_info=True)
        # Return an error structure so the DSE can continue
        return {"error": str(e)}
    finally:
        if mlflow.active_run():
            mlflow.end_run()
            logging.info(f"Ended MLflow run: {mlflow_run_name}")