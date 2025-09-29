# privateer_ad/marimo/marimo.py

import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import logging
    from pathlib import Path

    import numpy as np
    import os

    from privateer_ad.config import DataConfig, PathConfig, MetadataConfig
    from privateer_ad.old_data_utils import OldDataProcessor
    from privateer_ad.alveo.alveo_runner import AlveoRunner, AlveoRunnerParameters
    from privateer_ad.evaluate.evaluator_alveo import AlveoEvaluator
    import torch

    from privateer_ad.architectures import TransformerAD
    from privateer_ad.config import ModelConfig
    from privateer_ad.evaluate.evaluator import ModelEvaluator
    from privateer_ad.utils import load_model_weights

    logging.basicConfig(level=logging.INFO)
    mo.md("### 1) Imports ready")
    USE_OLD_METADATA = True
    return (
        AlveoEvaluator,
        AlveoRunner,
        AlveoRunnerParameters,
        DataConfig,
        OldDataProcessor,
        PathConfig,
        USE_OLD_METADATA,
        logging,
        mo,
        os,
    )


@app.cell
def _(DataConfig, OldDataProcessor, PathConfig, USE_OLD_METADATA, logging, mo):
    mo.md("### 2) Build data pipeline (test)")
    dconf = DataConfig(seq_len=12)  # override here if you want: DataConfig(seq_len=..., batch_size=...)
    # 1) Initialize old data processor
    dp = OldDataProcessor(use_old_metadata=USE_OLD_METADATA, recompute_attack_from_metadata=USE_OLD_METADATA)    # 2) Ensure processed splits exist; if not, build them and fit scalers
    pc = PathConfig()
    train_csv = pc.processed_dir.joinpath("train.csv")
    val_csv = pc.processed_dir.joinpath("val.csv")
    test_csv = pc.processed_dir.joinpath("test.csv")
    if not (train_csv.exists() and val_csv.exists() and test_csv.exists()):
        # This will split per device (stratified) and fit scaler (and PCA if you pass n_components)
        dp.initialize_data_pipeline(dataset_path=pc.raw_dataset, train_size=0.8)

    # 3) Build a test dataloader with old defaults (seq_len=12, batch_size=4096)
    test_dl= dp.get_dataloader(
        split="test",
        use_pca=False,        # flip to True if you fitted PCA in initialize_data_pipeline(n_components=...)
        batch_size=4096,
        seq_len=12,
        partition_id=None,    # or an int if you want a federated-like partition
        only_benign=False,
        num_workers=16,
    )

    logging.info(
        f"Data ready: seq_len={dconf.seq_len}, batch_size={dconf.batch_size}, features={len(dp.input_features)}"
    )
    len(test_dl)  # small peek makes Marimo display something
    return dconf, dp, test_dl


@app.cell
def _(mo):
    mo.md("### 3) Check Alveo devices (pynq)")
    try:
        import pynq
        device_list = list(pynq.Device.devices)
        for i, dev in enumerate(device_list):
            print(f"{i}) {dev.name}")
    except Exception as e:
        device_list = []
        print(f"Could not enumerate PYNQ devices. Reason: {e}")
        print("If you're not on the FPGA machine yet, this is expected.")
    device_list
    return (device_list,)


@app.cell
def _(
    AlveoRunner,
    AlveoRunnerParameters,
    PathConfig,
    dconf,
    device_list,
    dp,
    mo,
    os,
):
    mo.md("### 4) Create Alveo runner")
    XCLBIN_PATH = os.path.join(PathConfig().experiments_dir, "alveo_xclbins", "attention_ae_fxp32.xclbin")

    # Bus can be None if you only want overlay + kernel (no power scraping)
    DEVICE_BUS = None  # e.g. "0000:af:00.1"
    DEVICE_INDEX = 0 if device_list else 0  # default 0 if available

    n_features = len(dp.input_features)
    params = AlveoRunnerParameters(
        input_buffer_elements=dconf.seq_len * n_features,
        output_buffer_elements=dconf.seq_len * n_features,
        kernel_name=None,  # set if your overlay has a named IP block
    )

    device = device_list[DEVICE_INDEX] if device_list else None
    try:
        runner = AlveoRunner(
            bitstream_path=XCLBIN_PATH,
            parameters=params,
            device=device,
            device_bus=DEVICE_BUS,
        )
        print("Runner initialized successfully.")
    except Exception as e:
        runner = None
        print(f"Failed to initialize AlveoRunner: {e}")

    XCLBIN_PATH, DEVICE_BUS, DEVICE_INDEX, runner
    return (runner,)


@app.cell
def _(AlveoEvaluator, mo, runner, test_dl):
    mo.md("### 5) Evaluate on Alveo")
    try:
        if runner is None:
            raise RuntimeError("AlveoRunner is not initialized (see cell 4).")

        evaluator = AlveoEvaluator()
        metrics, figures = evaluator.evaluate(
            runner=runner,
            dataloader=test_dl,
            threshold=0.025,
            prefix="alveo",
            step=0,
        )
        print("== Alveo Evaluation Metrics ==")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        print("\nFigures available in `figures` dict (matplotlib Figure objects).")
    except Exception as e:
        metrics, figures = {}, {}
        print(f"Evaluation failed: {e}")
    metrics, figures
    return


@app.cell
def _(runner):
    runner.clean_class()
    return


if __name__ == "__main__":
    app.run()
