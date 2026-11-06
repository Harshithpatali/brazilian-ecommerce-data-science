from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from app.components.ui import hero, metric_card, section_title
from src.features.churn_features import FEATURE_COLS, add_churn_features
from src.models.model_contract import validate_model_metadata

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "ml" / "customer_churn_dataset.parquet"
MODEL_PATH = ROOT / "models" / "best_churn_model.joblib"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
COMPARISON_PATH = ROOT / "models" / "model_comparison.csv"


@st.cache_data
def load_features() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Build the SQL churn dataset and run scripts.build_ml_data.")
    return pd.read_parquet(DATA_PATH)


@st.cache_resource
def load_model(model_path: str, expected_sha256: str):
    path = Path(model_path)
    if not path.exists():
        return None
    actual_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError("Model binary SHA-256 does not match model_metadata.json.")
    return joblib.load(path)


@st.cache_data
def load_metadata() -> dict:
    if not METADATA_PATH.exists():
        return {}
    return json.loads(METADATA_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_comparison() -> pd.DataFrame:
    return pd.read_csv(COMPARISON_PATH) if COMPARISON_PATH.exists() else pd.DataFrame()


def render() -> None:
    hero(
        "Customer Churn Intelligence",
        "Historical point-in-time customer scoring using a 90-day forward inactivity target.",
    )
    try:
        metadata = load_metadata()
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read churn model metadata: {exc}")
        return

    contract_errors = validate_model_metadata(metadata)
    if contract_errors:
        st.warning(
            "The churn model artifacts are missing, stale, or inconsistent. "
            "Rebuild the PostgreSQL churn dataset, run the corrected training "
            "pipeline, and redeploy the new model artifacts."
        )
        for error in contract_errors:
            st.write(f"- {error}")
        st.code(
            "python -m scripts.build_all\n"
            "python -m scripts.validate_churn_data\n"
            "python -m scripts.run_ml",
            language="bash",
        )
        return

    try:
        model = load_model(str(MODEL_PATH), metadata["model_sha256"])
    except (OSError, ValueError, AttributeError) as exc:
        st.error(f"Could not load the selected model artifact: {exc}")
        return
    if model is None:
        st.warning("The selected model artifact is missing. Run the training pipeline.")
        return

    try:
        df = load_features().copy()
    except (OSError, ValueError) as exc:
        st.error(f"Could not load the historical churn scoring dataset: {exc}")
        return
    if df.empty:
        st.info("No historical churn snapshots are available.")
        return
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df = df.dropna(subset=["snapshot_date"])
    latest_date = df["snapshot_date"].max()
    latest = df[df["snapshot_date"] == latest_date].copy()
    if latest.empty:
        st.info("No latest historical snapshot is available.")
        return

    features = add_churn_features(latest)
    missing = sorted(set(FEATURE_COLS).difference(features.columns))
    if missing:
        st.error(f"Scoring dataset is missing features: {missing}")
        return
    try:
        latest["churn_probability"] = model.predict_proba(features[FEATURE_COLS])[:, 1]
    except (ValueError, AttributeError, KeyError) as exc:
        st.error(f"Could not score the latest historical snapshot: {exc}")
        return

    threshold = float(metadata["threshold"])
    high_risk = latest[latest["churn_probability"] >= threshold].copy()
    st.info(
        "This is a historical Olist dataset demonstration, not a live customer feed. "
        "Predictions refer to the latest snapshot available in the sample data."
    )

    columns = st.columns(4)
    cards = [
        ("Selected Model", metadata.get("selected_model", "N/A"), "Selected by temporal CV PR-AUC", "blue"),
        ("Latest Snapshot", str(latest_date.date()), "Historical prediction population", "violet"),
        ("Average Predicted Risk", f"{latest['churn_probability'].mean():.1%}", "Latest snapshot", "teal"),
        ("Customers Above Threshold", f"{len(high_risk):,}", f"Risk ≥ {threshold:.1%}", "amber"),
    ]
    for column, (label, value, help_text, accent) in zip(columns, cards):
        with column:
            metric_card(label, value, help_text=help_text, accent=accent)

    test_metrics = metadata["test_metrics"]
    baseline = metadata.get("dummy_baseline", {})
    section_title(
        "Final chronological test evaluation",
        "The selected model and threshold were frozen before this final holdout was evaluated.",
    )
    metric_columns = st.columns(4)
    values = [
        ("Test PR-AUC", test_metrics.get("pr_auc"), "Average precision"),
        ("Dummy PR-AUC", baseline.get("test_pr_auc"), "Prior-probability baseline"),
        ("Test ROC-AUC", test_metrics.get("roc_auc"), "Threshold independent"),
        ("Test F1", test_metrics.get("f1"), f"Validation threshold {threshold:.3f}"),
    ]
    for column, (label, value, help_text) in zip(metric_columns, values):
        with column:
            metric_card(label, "N/A" if value is None else f"{float(value):.3f}", help_text=help_text)
    prevalence = baseline.get("test_positive_prevalence")
    if prevalence is not None:
        st.caption(f"Test positive prevalence: {float(prevalence):.1%}. Interpret PR-AUC relative to this baseline.")
    if test_metrics.get("pr_auc_lift_over_dummy") is not None:
        st.caption(
            f"PR-AUC lift over the prior-probability baseline: "
            f"{float(test_metrics['pr_auc_lift_over_dummy']):+.4f}"
        )

    section_title(
        "Class balance by partition",
        "Counts are shown separately for train, validation and final test. "
        "Class 1 is the 90-day churn label.",
    )
    distribution = metadata.get("class_distribution", {})
    distribution_rows = []
    for partition in ["train", "validation", "test"]:
        counts = distribution.get(partition, {})
        class_0 = int(counts.get("0", 0))
        class_1 = int(counts.get("1", 0))
        total = class_0 + class_1
        distribution_rows.append({
            "Partition": partition.title(),
            "Purchase within 90 days (0)": class_0,
            "No delivered purchase (1)": class_1,
            "Churn prevalence": class_1 / total if total else None,
        })
    st.dataframe(
        pd.DataFrame(distribution_rows),
        use_container_width=True,
        hide_index=True,
    )

    confusion_path = ROOT / "models" / "test_confusion_matrix.json"
    if confusion_path.exists():
        confusion = json.loads(confusion_path.read_text(encoding="utf-8"))
        matrix = confusion.get("matrix", [])
        labels = confusion.get("labels", ["purchase_within_90_days", "churned"])
        if len(matrix) == 2 and all(len(row) == 2 for row in matrix):
            section_title(
                "Final test confusion matrix",
                "Rows are actual classes and columns are predicted classes, "
                f"at threshold {float(confusion.get('threshold', threshold)):.3f}.",
            )
            confusion_frame = pd.DataFrame(
                matrix,
                index=[f"Actual: {label}" for label in labels],
                columns=[f"Predicted: {label}" for label in labels],
            )
            st.dataframe(confusion_frame, use_container_width=True)

    pr_curve_path = ROOT / "models" / "test_precision_recall_curve.csv"
    if pr_curve_path.exists():
        pr_curve = pd.read_csv(pr_curve_path)
        if {"threshold", "precision", "recall"}.issubset(pr_curve.columns) and not pr_curve.empty:
            section_title(
                "Final test precision-recall curve",
                "This curve is generated from the final holdout. Use it to inspect "
                "the precision/recall trade-off across thresholds.",
            )
            pr_curve = pr_curve.sort_values("threshold")
            st.line_chart(
                pr_curve.set_index("threshold")[["precision", "recall"]],
                use_container_width=True,
            )

    comparison = load_comparison()
    section_title(
        "Candidate model comparison",
        "Models are ranked by expanding-window cross-validation PR-AUC; the final test set is not used for selection.",
    )
    if not comparison.empty:
        show = [c for c in ["model", "cv_pr_auc", "selected", "best_params"] if c in comparison.columns]
        st.dataframe(comparison[show], use_container_width=True, hide_index=True)

    section_title(
        "Highest-risk customers",
        "Latest historical snapshot ranked by probability of no delivered purchase in the next 90 days.",
    )
    display_cols = [
        "customer_unique_id", "churn_probability", "orders_180d", "revenue_180d",
        "recency_days", "avg_review_score_180d", "avg_delivery_delay_180d",
    ]
    shown = high_risk.sort_values("churn_probability", ascending=False)[
        [c for c in display_cols if c in high_risk.columns]
    ].head(100).copy()
    if not shown.empty:
        shown["churn_probability"] = shown["churn_probability"].map(lambda x: f"{x:.1%}")
    st.dataframe(shown, use_container_width=True, hide_index=True)

    section_title(
        "Intervention threshold",
        "Default threshold maximizes F1 on the separate validation period. Changing this control only changes the displayed customer count.",
    )
    threshold_view = st.slider(
        "Probability threshold", min_value=0.01, max_value=0.99,
        value=float(max(0.01, min(0.99, threshold))), step=0.01,
    )
    count = int((latest["churn_probability"] >= threshold_view).sum())
    metric_card("Customers above selected threshold", f"{count:,}", help_text=f"Threshold {threshold_view:.1%}")

    with st.expander("Evaluation and methodology"):
        st.write(metadata.get("target_definition", ""))
        st.write(metadata.get("evaluation_design", ""))
        st.write(
            "F1-based threshold selection is a technical proxy, not a business-optimal "
            "retention policy until intervention costs and customer value are specified."
        )
        st.json({
            "model_version": metadata.get("model_version"),
            "selected_model": metadata.get("selected_model"),
            "threshold": metadata.get("threshold"),
            "periods": metadata.get("periods", {}),
            "class_distribution": metadata.get("class_distribution", {}),
            "test_metrics": test_metrics,
            "dummy_baseline": baseline,
            "best_params": metadata.get("best_params", {}),
            "features": metadata.get("feature_columns", []),
        })
