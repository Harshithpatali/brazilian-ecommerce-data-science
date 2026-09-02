from __future__ import annotations

import os
from pathlib import Path

import mlflow
from dotenv import load_dotenv

load_dotenv()


def configure_mlflow() -> str:
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "").strip()
    if not tracking_uri:
        project_root = Path(__file__).resolve().parents[2]
        # Convert Windows paths (e.g. D:\\project\\mlruns) to a valid
        # MLflow file URI (e.g. file:///D:/project/mlruns).
        tracking_uri = (project_root / "mlruns").resolve().as_uri()

    experiment_name = os.getenv(
        "MLFLOW_EXPERIMENT_NAME",
        "brazilian-ecommerce-churn",
    )

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return experiment_name


def get_tracking_uri() -> str:
    return mlflow.get_tracking_uri()
