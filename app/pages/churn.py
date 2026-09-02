from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from app.components.ui import hero, metric_card, section_title
from src.features.churn_features import FEATURE_COLS, add_churn_features

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "ml" / "customer_churn_dataset.parquet"
MODEL_PATH = ROOT / "models" / "best_churn_model.joblib"
METRICS_PATH = ROOT / "models" / "model_metadata.json"
COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"


@st.cache_data

def load_features() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "ML dataset not found. Run `python -m scripts.run_ml`."
        )
    return pd.read_parquet(DATA_PATH)


@st.cache_resource

def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data

def load_metadata() -> dict:
    if not METRICS_PATH.exists():
        return {}
    return json.loads(METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data

def load_comparison() -> pd.DataFrame:
    if not COMPARISON_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(COMPARISON_PATH)


def render() -> None:
    hero(
        "Customer Churn Intelligence",
        "Tuned machine-learning models scored against point-in-time customer behavior.",
    )

    model = load_model()
    metadata = load_metadata()
    comparison = load_comparison()
    if model is None or not metadata:
        st.warning(
            "The tuned model has not been trained yet. Run `python -m scripts.run_ml` "
            "from the project root."
        )
        return

    df = load_features().copy()
    if df.empty:
        st.info("No ML data is available.")
        return

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    latest_date = df["snapshot_date"].max()
    latest = df[df["snapshot_date"] == latest_date].copy()
    if latest.empty:
        st.info("No latest snapshot is available.")
        return

    latest_features = add_churn_features(latest)
    latest["churn_probability"] = model.predict_proba(
        latest_features[FEATURE_COLS]
    )[:, 1]

    threshold = float(metadata.get("threshold", 0.50))
    high_risk = latest[latest["churn_probability"] >= threshold].copy()
    avg_risk = latest["churn_probability"].mean()

    cols = st.columns(4)
    vals = [
        ("Best Model", metadata.get("selected_model", "N/A"), "Selected by PR-AUC", "blue"),
        ("Latest Snapshot", str(latest_date.date()), "Prediction population", "violet"),
        ("Avg Churn Risk", f"{avg_risk:.1%}", "Latest snapshot", "teal"),
        ("High-Risk Customers", f"{len(high_risk):,}", f"Risk ≥ {threshold:.0%}", "amber"),
    ]
    for col, (label, value, help_text, accent) in zip(cols, vals):
        with col:
            metric_card(label, value, help_text=help_text, accent=accent)

    section_title(
        "Model benchmark",
        "Two classifiers were tuned with expanding-window temporal cross-validation. The latest snapshots remain an untouched test period.",
    )
    if not comparison.empty:
        show_cols = [
            "model", "cv_pr_auc", "pr_auc", "roc_auc",
            "precision", "recall", "f1",
        ]
        available = [c for c in show_cols if c in comparison.columns]
        st.dataframe(
            comparison[available].round(4),
            use_container_width=True,
            hide_index=True,
        )

    section_title(
        "Highest-risk customers",
        f"Latest snapshot ranked by predicted probability of no delivered purchase in the next 90 days. Default intervention threshold: {threshold:.0%}.",
    )
    display_cols = [
        "customer_unique_id",
        "churn_probability",
        "orders_180d",
        "revenue_180d",
        "recency_days",
        "avg_review_score_180d",
        "avg_delivery_delay_180d",
    ]
    display = high_risk.sort_values("churn_probability", ascending=False)
    shown = display[[c for c in display_cols if c in display.columns]].head(100).copy()
    if not shown.empty:
        shown["churn_probability"] = shown["churn_probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(shown, use_container_width=True, hide_index=True)

    section_title("Risk threshold", "Adjust the intervention threshold without retraining the model.")
    threshold_view = st.slider(
        "Probability threshold",
        min_value=0.50,
        max_value=0.95,
        value=float(max(0.50, min(0.95, threshold))),
        step=0.05,
    )
    above = latest[latest["churn_probability"] >= threshold_view]
    metric_card("Customers above selected threshold", f"{len(above):,}", accent="rose")

    with st.expander("Model methodology"):
        st.write(
            "The model uses leakage-safe features calculated strictly before each snapshot. "
            "The target is 1 when the customer makes no delivered purchase during the following 90 days. "
            "The final test period is chronologically after the training period. Three models are tuned: "
            "Logistic Regression, Random Forest, and HistGradientBoosting. Hyperparameters are selected on "
            "temporal cross-validation using PR-AUC because churn is an imbalanced classification problem."
        )
        st.json({
            "selected_model": metadata.get("selected_model"),
            "threshold": metadata.get("threshold"),
            "features": metadata.get("feature_columns", []),
            "test_metrics": metadata.get("test_metrics", {}),
            "best_params": metadata.get("best_params", {}),
        })
