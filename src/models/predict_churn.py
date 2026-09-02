from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from src.features.churn_features import FEATURE_COLS, add_churn_features

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "best_churn_model.joblib"


def load_best_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError("Best churn model not found. Run `python -m scripts.run_ml`.")
    return joblib.load(MODEL_PATH)


def predict_proba(df: pd.DataFrame) -> pd.Series:
    features = add_churn_features(df)
    model = load_best_model()
    return pd.Series(
        model.predict_proba(features[FEATURE_COLS])[:, 1],
        index=df.index,
        name="churn_probability",
    )
