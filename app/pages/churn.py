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
        raise FileNotFoundError("Build the SQL repeat-propensity dataset and run scripts.build_ml_data.")
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
        "Customer Repeat-Purchase Intelligence",
        "Monthly CRM propensity ranking: which active customers are most likely to make a delivered purchase in the next 90 days?",
    )
    try:
        metadata = load_metadata()
    except (OSError, json.JSONDecodeError) as exc:
        st.error(f"Could not read repeat-propensity metadata: {exc}")
        return

    contract_errors = validate_model_metadata(metadata)
    if contract_errors:
        st.warning(
            "The v5 propensity artifacts are missing, stale, or inconsistent. "
            "Rebuild the PostgreSQL dataset, run the v5 training pipeline, and redeploy."
        )
        for error in contract_errors:
            st.write(f"- {error}")
        st.code(
            "python -m scripts.refresh_ml_sql\n"
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
        st.warning("The selected propensity model artifact is missing.")
        return

    try:
        df = load_features().copy()
    except (OSError, ValueError) as exc:
        st.error(f"Could not load the historical propensity dataset: {exc}")
        return
    if df.empty:
        st.info("No historical customer snapshots are available.")
        return

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df = df.dropna(subset=["snapshot_date", "customer_unique_id"])
    latest_date = df["snapshot_date"].max()
    latest = df[df["snapshot_date"] == latest_date].copy()
    if latest.empty:
        st.info("No latest monthly customer population is available.")
        return

    features = add_churn_features(latest)
    missing = sorted(set(FEATURE_COLS).difference(features.columns))
    if missing:
        st.error(f"Scoring dataset is missing model features: {missing}")
        return
    try:
        probabilities = model.predict_proba(features[FEATURE_COLS])
        latest["repeat_purchase_probability"] = probabilities[:, 0]
        latest["no_purchase_probability"] = probabilities[:, 1]
    except (ValueError, AttributeError, KeyError) as exc:
        st.error(f"Could not score the latest monthly population: {exc}")
        return

    st.info(
        "Historical Olist demonstration: these are backfilled predictions for the "
        "latest available monthly snapshot, not a live marketplace feed."
    )

    default_capacity = int(round(float(metadata.get("capacity_fraction", 0.10)) * 100))
    capacity_options = [5, 10, 20]
    selected_capacity = st.select_slider(
        "Monthly campaign contact capacity",
        options=capacity_options,
        value=default_capacity if default_capacity in capacity_options else 10,
        help="The app ranks customers within the latest snapshot and selects the highest propensity customers up to this campaign capacity.",
    )
    priority_count = max(1, int(len(latest) * selected_capacity / 100 + 0.999999))
    priority = latest.sort_values(
        "repeat_purchase_probability", ascending=False, kind="mergesort"
    ).head(priority_count).copy()

    columns = st.columns(4)
    cards = [
        ("Selected Model", metadata.get("selected_model", "N/A"), "Selected by purged temporal-CV PR-AUC", "blue"),
        ("Latest Snapshot", str(latest_date.date()), "Historical monthly decision date", "violet"),
        ("Active Customers", f"{len(latest):,}", "Eligible and active within prior 180 days", "teal"),
        ("Campaign Priority List", f"{len(priority):,}", f"Top {selected_capacity}% by calibrated propensity", "amber"),
    ]
    for column, (label, value, help_text, accent) in zip(columns, cards):
        with column:
            metric_card(label, value, help_text=help_text, accent=accent)

    test_metrics = metadata["test_metrics"]
    baseline = metadata.get("dummy_baseline", {})
    section_title(
        "Final chronological holdout — ranking and calibration",
        "The base model was selected by purged temporal CV; probabilities were calibrated on the separate validation period. The final test period was not used to fit either.",
    )
    metric_columns = st.columns(4)
    values = [
        ("Repeat-purchase PR-AUC", test_metrics.get("pr_auc"), "Minority-event average precision"),
        ("Prior baseline PR-AUC", baseline.get("test_pr_auc"), "Training-prevalence baseline"),
        ("Recency-only PR-AUC", baseline.get("recency_only_pr_auc"), "Simple operational benchmark"),
        ("ROC-AUC", test_metrics.get("roc_auc"), "Threshold independent ranking"),
    ]
    for column, (label, value, help_text) in zip(metric_columns, values):
        with column:
            metric_card(label, "N/A" if value is None else f"{float(value):.3f}", help_text=help_text)

    section_title(
        "Campaign-capacity performance",
        "Each monthly snapshot is ranked separately. Precision is the share of contacted customers who purchase within 90 days; recall is the share of all repeat purchasers reached.",
    )
    capacity_rows = []
    for fraction, label in [("5pct", "Top 5%"), ("10pct", "Top 10%"), ("20pct", "Top 20%")]:
        capacity_rows.append({
            "Campaign list": label,
            "Precision": test_metrics[f"precision_at_{fraction}"],
            "Recall": test_metrics[f"recall_at_{fraction}"],
            "Lift vs prevalence": test_metrics[f"lift_at_{fraction}"],
            "Flagged share": test_metrics[f"flagged_fraction_at_{fraction}"],
        })
    capacity_table = pd.DataFrame(capacity_rows)
    st.dataframe(
        capacity_table.style.format({
            "Precision": "{:.2%}",
            "Recall": "{:.2%}",
            "Lift vs prevalence": "{:.2f}x",
            "Flagged share": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    bootstrap = metadata["customer_bootstrap_pr_auc_95_ci"]
    st.caption(
        f"Customer-cluster bootstrap 95% CI for PR-AUC: "
        f"{bootstrap['lower_95']:.4f}–{bootstrap['upper_95']:.4f} "
        f"({bootstrap['replicates']} valid replicates)."
    )
    st.caption(
        f"PR-AUC vs prior baseline: {test_metrics['pr_auc_lift_over_dummy']:+.4f}; "
        f"PR-AUC / recency-only AP: {test_metrics['pr_auc_lift_over_recency']:.2f}x."
    )

    calibration_columns = st.columns(3)
    with calibration_columns[0]:
        metric_card("Brier score", f"{test_metrics['brier_score']:.4f}", help_text="Lower is better for calibrated probabilities")
    with calibration_columns[1]:
        metric_card("Log loss", f"{test_metrics['log_loss']:.4f}", help_text="Lower is better")
    with calibration_columns[2]:
        metric_card("Brier skill score", f"{test_metrics['brier_skill_score']:.3f}", help_text="Positive means better than the prevalence-only forecast")

    calibration_path = ROOT / "models" / "test_calibration.csv"
    if calibration_path.exists():
        calibration = pd.read_csv(calibration_path)
        if {"mean_predicted_probability", "observed_repeat_rate"}.issubset(calibration.columns):
            section_title(
                "Holdout calibration",
                "Predicted repeat-purchase probabilities are grouped into deciles; the observed rate should track the mean forecast.",
            )
            chart = calibration.set_index("probability_decile")[
                ["mean_predicted_probability", "observed_repeat_rate"]
            ]
            st.line_chart(chart, use_container_width=True)

    section_title(
        "Class-level metrics at the top-10% monthly policy",
        "Thresholded reporting uses a fixed capacity policy rather than a global F2 cutoff.",
    )
    class_report = pd.DataFrame([
        {
            "Event": "Purchase within 90 days",
            "Precision": test_metrics["class_0_precision"],
            "Recall": test_metrics["class_0_recall"],
            "F1": test_metrics["class_0_f1"],
        },
        {
            "Event": "No purchase within 90 days",
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
    quality_columns = st.columns(2)
    with quality_columns[0]:
        metric_card("Balanced accuracy", f"{test_metrics['balanced_accuracy']:.3f}", help_text="Average recall across both classes")
    with quality_columns[1]:
        metric_card("Matthews correlation", f"{test_metrics['matthews_corrcoef']:.3f}", help_text="Correlation using all confusion-matrix cells")

    section_title(
        "Class distribution",
        "Counts are customer-month decisions, not independent unique customers. The customer-bootstrap interval accounts for repeated customer observations.",
    )
    distribution = metadata.get("class_distribution", {})
    distribution_rows = []
    for partition in ["train", "validation", "test"]:
        counts = distribution.get(partition, {})
        repeat = int(counts.get("0", 0))
        no_repeat = int(counts.get("1", 0))
        total = repeat + no_repeat
        distribution_rows.append({
            "Partition": partition.title(),
            "Repeat purchase (0)": repeat,
            "No purchase (1)": no_repeat,
            "Repeat rate": repeat / total if total else None,
        })
    st.dataframe(pd.DataFrame(distribution_rows), use_container_width=True, hide_index=True)

    confusion_path = ROOT / "models" / "test_confusion_matrix.json"
    if confusion_path.exists():
        confusion = json.loads(confusion_path.read_text(encoding="utf-8"))
        matrix = confusion.get("matrix", [])
        labels = confusion.get("labels", ["purchase_within_90_days", "no_purchase_within_90_days"])
        if len(matrix) == 2 and all(len(row) == 2 for row in matrix):
            section_title(
                "Final test confusion matrix",
                "Rows are actual events and columns are predictions under the top-10% per-month contact policy.",
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
        if {
            "repeat_purchase_probability_threshold", "precision", "recall"
        }.issubset(pr_curve.columns) and not pr_curve.empty:
            section_title(
                "Final test precision-recall curve",
                "This curve comes from the final holdout. The campaign policy is selected separately by monthly capacity.",
            )
            pr_curve = pr_curve.sort_values("repeat_purchase_probability_threshold")
            st.line_chart(
                pr_curve.set_index("repeat_purchase_probability_threshold")[["precision", "recall"]],
                use_container_width=True,
            )

    comparison = load_comparison()
    section_title(
        "Candidate model comparison",
        "Candidates are ranked by purged expanding-window temporal-CV average precision for repeat purchase; the final test set is not used for model selection.",
    )
    if not comparison.empty:
        show = [
            column for column in
            ["model", "cv_repeat_purchase_pr_auc", "selected", "best_params"]
            if column in comparison.columns
        ]
        st.dataframe(comparison[show], use_container_width=True, hide_index=True)

    section_title(
        "Priority customers",
        f"Highest calibrated repeat-purchase propensity in the latest historical snapshot. "
        f"Showing the top {selected_capacity}% campaign list.",
    )
    display_cols = [
        "customer_unique_id",
        "customer_state",
        "repeat_purchase_probability",
        "orders_180d",
        "revenue_180d",
        "recency_days",
        "avg_review_score_180d",
        "avg_delivery_delay_180d",
    ]
    shown = priority[
        [column for column in display_cols if column in priority.columns]
    ].copy()
    if "repeat_purchase_probability" in shown.columns:
        shown["repeat_purchase_probability"] = shown[
            "repeat_purchase_probability"
        ].map(lambda value: f"{value:.1%}")
    st.dataframe(shown, use_container_width=True, hide_index=True)

    with st.expander("Evaluation design and intended use"):
        st.write(metadata.get("population_definition", ""))
        st.write(metadata.get("target_definition", ""))
        st.write(metadata.get("evaluation_design", ""))
        st.write(
            "This is historical offline evaluation. Do not contact customers "
            "solely because of a model score until a controlled campaign test "
            "demonstrates incremental value and operational suitability."
        )
        st.json({
            "model_version": metadata.get("model_version"),
            "selected_model": metadata.get("selected_model"),
            "threshold_policy": metadata.get("threshold_policy"),
            "capacity_fraction": metadata.get("capacity_fraction"),
            "calibration_method": metadata.get("calibration_method"),
            "periods": metadata.get("periods", {}),
            "partition_rows": metadata.get("partition_rows", {}),
            "unique_customers": metadata.get("unique_customers", {}),
            "test_metrics": test_metrics,
            "customer_bootstrap_pr_auc_95_ci": bootstrap,
            "best_params": metadata.get("best_params", {}),
            "features": metadata.get("feature_columns", []),
        })
