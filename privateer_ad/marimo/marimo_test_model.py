# privateer_ad/marimo/marimo_test_model.py

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import logging
    from pathlib import Path
    import torch
    import mlflow
    from privateer_ad.config import (
        DataConfig,
        ModelConfig,
        PathConfig
    )
    from torchinfo import summary
    from privateer_ad.etl.transform import DataProcessor
    from privateer_ad.architectures import TransformerAD
    from privateer_ad.evaluate.evaluator import ModelEvaluator

    logging.basicConfig(level=logging.INFO)
    mo.md("### 1) Imports and setup complete")
    return (
        DataConfig,
        DataProcessor,
        ModelEvaluator,
        Path,
        PathConfig,
        logging,
        mlflow,
        mo,
        summary,
        torch,
    )


@app.cell
def _(DataConfig, DataProcessor, logging, mo):
    mo.md("### 2) Prepare test dataloader")
    data_config = DataConfig(seq_len=77, batch_size=4096)
    dp = DataProcessor(data_config=data_config)
    test_dl = dp.get_dataloader("test", only_benign=False, train=False)
    logging.info(f"Test dataloader ready with {len(test_dl)} batches")
    return (test_dl,)


@app.cell
def _(Path, PathConfig, logging, mlflow, mo, summary, test_dl):
    mo.md("### 3) Load MLflow-logged model from local experiments directory")
    # Point to the full MLflow model directory (not just model.pth!)
    model_dir = Path(PathConfig().root_dir) / "experiments" / "TransformerAD_DP_v15" / "TransformerAD_DP"
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    model_uri = f"file://{model_dir.resolve()}"
    logging.info(f"Loading MLflow model from: {model_uri}")

    # Load model directly from local MLflow artifact
    model = mlflow.pytorch.load_model(model_uri, map_location='cpu')
    model.eval()
    logging.info("✅ Model loaded successfully from MLflow artifact")

    ## Showcase the models structure
    sample = next(iter(test_dl))[0]['encoder_cont'][:1]
    model_summary = summary(model,
                            input_data=sample,
                            col_names=('input_size', 'output_size', 'num_params', 'params_percent'))
    model_summary_deep = summary(model,
                            input_data=sample,
                            col_names=('input_size', 'output_size', 'num_params', 'params_percent'),
                            depth=10)
    return (model,)


@app.cell
def _(ModelEvaluator, logging, mo, model, test_dl, torch):
    mo.md("### 4) Evaluate model on test set")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #model = model.to(device)

    evaluator = ModelEvaluator(device=device, loss_fn="L1Loss")
    metrics, figures = evaluator.evaluate(
        model=model,
        dataloader=test_dl,
        prefix="test",
        step=0
    )

    mo.md(f"### Evaluation complete! ROC-AUC: {metrics.get('test_roc_auc', 'N/A'):.4f}")
    logging.info("Evaluation metrics computed and visualizations generated.")
    return (figures,)


@app.cell
def _(figures, mo):
    mo.md("### 5) Visualizations")
    figures
    return


if __name__ == "__main__":
    app.run()
