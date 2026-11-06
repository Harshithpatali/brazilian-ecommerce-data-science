from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_module(module: str) -> None:
    subprocess.run([sys.executable, "-m", module], check=True, cwd=ROOT)


if __name__ == "__main__":
    print("1/2 Exporting the v5 first-delivery active-customer propensity dataset...")
    run_module("scripts.build_ml_data")
    print("2/2 Tuning four propensity models, calibrating on validation and tracking in MLflow...")
    run_module("src.models.train_churn")
    print("\nRepeat-purchase propensity v5 pipeline completed.")
    print("Launch MLflow UI with:")
    print("  python -m scripts.start_mlflow")
