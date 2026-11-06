from __future__ import annotations

import hashlib
import json
from math import ceil
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
        "Second-Purchase Propensity Intelligence",
        "Point-in-time ranking of customers after their first delivered order for a second purchase within 90 days.",
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

    latest["repeat_purchase_probability"] = 1.0 - latest["churn_probability"]
    threshold = float(metadata["threshold"])
    st.info(
        "This is a historical Olist dataset demonstration, not a live customer feed. "
        "Predictions refer to the latest snapshot available in the sample data."
    )

    columns = st.columns(4)
    cards = [
        ("Selected Model", metadata.get("selected_model", "N/A"), "Selected by temporal CV repeat-purchase PR-AUC", "blue"),
        ("Latest Snapshot", str(latest_date.date()), "Historical prediction population", "violet"),
        ("Average Repeat Probability", f"{latest['repeat_purchase_probability'].mean():.2%}", "Latest snapshot", "teal"),
        ("Eligible Customers", f"{len(latest):,}", "Latest historical cohort", "amber"),
    ]
    for column, (label, value, help_text, accent) in zip(columns, cards):
        with column:
            metric_card(label, value, help_text=help_text, accent=accent)

    test_metrics = metadata["test_metrics"]
    baseline = metadata.get("dummy_baseline", {})
    section_title(
        "Final chronological test evaluation",
        "The selected model and calibrator were frozen before the final holdout. Campaign capacity metrics rank customers within each monthly cohort.",
    )
    metric_columns = st.columns(4)
    values = [
        ("Repeat-purchase PR-AUC", test_metrics.get("pr_auc"), "Positive class: delivered purchase within 90 days"),
        ("Dummy repeat-purchase PR-AUC", baseline.get("test_pr_auc"), "Prior-probability baseline"),
        ("Repeat-purchase ROC-AUC", test_metrics.get("roc_auc"), "Threshold independent"),
        ("Repeat-purchase F2", test_metrics.get("f2"), "Validation-selected recall/precision trade-off"),

    ]
    for column, (label, value, help_text) in zip(metric_columns, values):
        with column:
            metric_card(label, "N/A" if value is None else f"{float(value):.3f}", help_text=help_text)

    section_title(
        "Per-class test performance",
        "Evaluation focuses on the minority event: class 0 (delivered purchase within 90 days). Class 1 is no delivered purchase.",
    )
    class_report = pd.DataFrame([
        {
            "Class": "Purchase within 90 days (0)",
            "Precision": test_metrics["class_0_precision"],
            "Recall": test_metrics["class_0_recall"],
            "F1": test_metrics["class_0_f1"],
        },
        {
            "Class": "No delivered purchase (1)",
            "Precision": test_metrics["class_1_precision"],
            "Recall": test_metrics["class_1_recall"],
            "F1": test_metrics["class_1_f1"],
        },
    ])
    st.dataframe(
        class_report.style.format({
            "Precision": "{:.3f}", "Recall": "{:.3f}", "F1": "{:.3f}"
        }),
        use_container_width=True,
        hide_index=True,
    )
    quality_columns = st.columns(4)
    with quality_columns[0]:
        metric_card(
            "Balanced accuracy",
            f"{test_metrics['balanced_accuracy']:.3f}",
            help_text="Average recall across both classes",
        )
    with quality_columns[1]:
        metric_card(
            "Matthews correlation",
            f"{test_metrics['matthews_corrcoef']:.3f}",
            help_text="Correlation-based score using all confusion-matrix cells",
        )
    with quality_columns[2]:
        metric_card(
            "Brier score",
            f"{test_metrics['brier_score']:.4f}",
            help_text="Lower is better; probability calibration",
        )
    with quality_columns[3]:
        metric_card(
            "Expected calibration error",
            f"{test_metrics['expected_calibration_error']:.4f}",
            help_text="Lower is better; 10 equal-width probability bins",
        )
    prevalence = baseline.get("test_positive_prevalence")
    if prevalence is not None:
        st.caption(f"Test repeat-purchase prevalence (SQL churned=0): {float(prevalence):.1%}. Interpret repeat-purchase PR-AUC relative to this baseline.")
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
        if {"repeat_purchase_probability_threshold", "precision", "recall"}.issubset(pr_curve.columns) and not pr_curve.empty:
            section_title(
                "Final test precision-recall curve",
                "This curve is generated from the final holdout. Use it to inspect "
                "the precision/recall trade-off across repeat-purchase probability thresholds.",
            )
            pr_curve = pr_curve.sort_values("repeat_purchase_probability_threshold")
            st.line_chart(
                pr_curve.set_index("repeat_purchase_probability_threshold")[["precision", "recall"]],
                use_container_width=True,
            )

    comparison = load_comparison()
    section_title(
        "Candidate model comparison",
        "Models are ranked by expanding-window cross-validation PR-AUC for the minority repeat-purchase event; the final test set is not used for selection.",
    )
    if not comparison.empty:
        show = [c for c in ["model", "cv_repeat_purchase_pr_auc", "selected", "best_params"] if c in comparison.columns]
        st.dataframe(comparison[show], use_container_width=True, hide_index=True)

    section_title(
        "Campaign capacity and ranking quality",
        "Precision@K and recall@K are measured by ranking customers separately within each monthly holdout cohort.",
    )
    capacity = metadata.get("test_capacity_metrics", {})
    baseline_capacity = metadata.get("baseline_capacity_metrics", {})
    capacity_rows = []
    for label in ("5%", "10%", "20%"):
        values = capacity.get(label, {})
        row = {
            "Monthly contact capacity": label,
            "Model precision": values.get("precision"),
            "Model recall": values.get("recall"),
            "Model lift": values.get("lift_over_monthly_prevalence"),
            "Model contacts": values.get("customers_contacted"),
        }
        for baseline_name, display_name in (
            ("recency_only", "Recency"),
            ("orders_180d_only", "Recent frequency"),
            ("lifetime_frequency_only", "Lifetime frequency"),
        ):
            row[f"{display_name} precision"] = baseline_capacity.get(
                baseline_name, {}
            ).get(label, {}).get("precision")
        capacity_rows.append(row)
    st.dataframe(
        pd.DataFrame(capacity_rows).style.format({
            "Model precision": "{:.2%}",
            "Model recall": "{:.2%}",
            "Model lift": "{:.2f}x",
            "Recency precision": "{:.2%}",
            "Recent frequency precision": "{:.2%}",
            "Lifetime frequency precision": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )
    ci = metadata.get("customer_cluster_bootstrap_pr_auc_ci95", {})
    st.caption(
        "Customer-cluster bootstrap 95% PR-AUC interval: "
        f"{float(ci.get('lower_95', 0.0)):.4f}–{float(ci.get('upper_95', 0.0)):.4f} "
        f"({ci.get('valid_replicates', 0)} valid resamples)."
    )
    unseen = metadata.get("unseen_customer_test_metrics", {})
    st.caption(
        "Unseen-customer test subset: "
        f"{unseen.get('customers', 0):,} customers, "
        f"{unseen.get('positive_count', 0):,} repeat-purchase events; "
        f"PR-AUC {float(unseen.get('pr_auc') or 0.0):.4f}."
    )

    section_title(
        "Latest repeat-purchase prospects",
        "Customers are ranked by calibrated second-purchase probability. A fixed capacity is more actionable than a global F2 threshold.",
    )
    capacity_view = st.slider(
        "Monthly campaign contact capacity",
        min_value=5, max_value=30, value=10, step=5,
        format="%d%%",
    )
    target_count = max(1, int(ceil(len(latest) * capacity_view / 100.0)))
    prospects = latest.nlargest(target_count, "repeat_purchase_probability").copy()
    display_cols = [
        "customer_unique_id", "repeat_purchase_probability",
        "lifetime_delivered_orders", "orders_180d", "revenue_180d",
        "recency_days", "avg_review_score_180d", "avg_delivery_delay_180d",
    ]
    shown = prospects[
        [column for column in display_cols if column in prospects.columns]
    ].copy()
    if not shown.empty:
        shown["repeat_purchase_probability"] = shown[
            "repeat_purchase_probability"
        ].map(lambda value: f"{value:.2%}")
    st.caption(
        f"Showing the top {capacity_view}% of the latest snapshot "
        f"({target_count:,} prospects). This is a historical demo, not a live customer feed."
    )
    st.dataframe(shown, use_container_width=True, hide_index=True)

    with st.expander("Evaluation and methodology"):
        st.write(metadata.get("target_definition", ""))
        st.write(metadata.get("evaluation_design", ""))
        st.write(f"PR-AUC positive class: {metadata.get('evaluation_positive_class', 'not specified')}")
        st.write(f"Repeat-purchase probability threshold: {metadata.get('repeat_purchase_probability_threshold', 'not available')}")
        st.write(
            "The F2 threshold is retained as a diagnostic operating point. For campaigns, use the top-K capacity ranking and agree a contact budget or minimum precision with the business."
        )
        st.json({
            "model_version": metadata.get("model_version"),
            "selected_model": metadata.get("selected_model"),
            "calibration": metadata.get("calibration", {}),
            "campaign_capacity_metrics": capacity,
            "threshold": metadata.get("threshold"),
            "periods": metadata.get("periods", {}),
            "class_distribution": metadata.get("class_distribution", {}),
            "test_metrics": test_metrics,
            "dummy_baseline": baseline,
            "best_params": metadata.get("best_params", {}),
            "features": metadata.get("feature_columns", []),
        })
