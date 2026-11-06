from __future__ import annotations

from datetime import date, timedelta
from math import isfinite
from typing import Any

from src.features.churn_features import CATEGORICAL_FEATURE_COLS, FEATURE_COLS

MODEL_VERSION = "churn-pit-v5"

_REQUIRED_METRICS = (
    "roc_auc",
    "gini",
    "ks_statistic",
    "expected_calibration_error",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "f2",
    "class_0_precision",
    "class_0_recall",
    "class_0_f1",
    "class_1_precision",
    "class_1_recall",
    "class_1_f1",
    "balanced_accuracy",
    "matthews_corrcoef",
    "positive_prevalence",
    "dummy_baseline_pr_auc",
    "pr_auc_lift_over_dummy",
    "recency_only_pr_auc",
    "pr_auc_lift_over_recency",
    "brier_score",
    "log_loss",
    "brier_skill_score",
    "precision_at_5pct",
    "recall_at_5pct",
    "lift_at_5pct",
    "flagged_fraction_at_5pct",
    "precision_at_10pct",
    "recall_at_10pct",
    "lift_at_10pct",
    "flagged_fraction_at_10pct",
    "precision_at_20pct",
    "recall_at_20pct",
    "lift_at_20pct",
    "flagged_fraction_at_20pct",
)

_BOUNDED_METRICS = (
    "roc_auc",
    "ks_statistic",
    "expected_calibration_error",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "f2",
    "class_0_precision",
    "class_0_recall",
    "class_0_f1",
    "class_1_precision",
    "class_1_recall",
    "class_1_f1",
    "balanced_accuracy",
    "positive_prevalence",
    "dummy_baseline_pr_auc",
    "recency_only_pr_auc",
    "brier_score",
    "precision_at_5pct",
    "recall_at_5pct",
    "flagged_fraction_at_5pct",
    "precision_at_10pct",
    "recall_at_10pct",
    "flagged_fraction_at_10pct",
    "precision_at_20pct",
    "recall_at_20pct",
    "flagged_fraction_at_20pct",
)


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and isfinite(float(value))


def validate_model_metadata(metadata: dict[str, Any]) -> list[str]:
    """Validate the v5 deployment and evaluation artifact contract."""
    if not isinstance(metadata, dict):
        return ["Model metadata must be a JSON object."]

    errors: list[str] = []
    if metadata.get("pipeline_status") != "ready":
        errors.append("pipeline_status must be 'ready'.")
    if metadata.get("model_version") != MODEL_VERSION:
        errors.append(
            f"model_version must be {MODEL_VERSION!r}; found "
            f"{metadata.get('model_version')!r}."
        )
    if metadata.get("feature_columns") != FEATURE_COLS:
        errors.append("feature_columns do not match the v5 application feature contract.")

    if metadata.get("evaluation_positive_class") != "purchase_within_90_days (SQL churned=0)":
        errors.append("evaluation_positive_class must identify purchase within 90 days.")

    if metadata.get("threshold_policy") != "top_10_percent_per_snapshot_date":
        errors.append("threshold_policy must be top_10_percent_per_snapshot_date.")
    capacity = metadata.get("capacity_fraction")
    if not _finite_number(capacity) or abs(float(capacity) - 0.10) > 1e-12:
        errors.append("capacity_fraction must be 0.10 for the v5 operational policy.")

    if metadata.get("calibration_method") != "Platt sigmoid fitted on separate chronological validation period":
        errors.append("calibration_method must document validation-only Platt calibration.")

    threshold = metadata.get("threshold")
    repeat_threshold = metadata.get("repeat_purchase_probability_threshold")
    if not _finite_number(threshold) or not 0 <= float(threshold) <= 1:
        errors.append("threshold must be finite and between 0 and 1.")
    if not _finite_number(repeat_threshold) or not 0 <= float(repeat_threshold) <= 1:
        errors.append("repeat_purchase_probability_threshold must be finite and between 0 and 1.")
    if _finite_number(threshold) and _finite_number(repeat_threshold):
        if abs(float(threshold) + float(repeat_threshold) - 1.0) > 1e-12:
            errors.append("Churn and repeat-purchase thresholds must be complementary.")

    availability = metadata.get("feature_availability")
    if not isinstance(availability, dict):
        errors.append("feature_availability is missing or is not an object.")
    else:
        expected = {
            "review_score": "review_answer_timestamp < snapshot_date",
            "lifetime_history": (
                "purchase_timestamp < snapshot_date and delivered_timestamp < snapshot_date"
            ),
            "prior_order_window": "[snapshot_date - 360 days, snapshot_date - 180 days)",
            "customer_state": "state from most recent delivered order observable before snapshot_date",
            "snapshot_month": "calendar month of snapshot_date",
            "future_orders": "used only to construct the target, never as predictors",
        }
        for name, value in expected.items():
            if availability.get(name) != value:
                errors.append(f"feature_availability.{name} does not match the v5 point-in-time definition.")

    model_hash = metadata.get("model_sha256")
    if (
        not isinstance(model_hash, str)
        or len(model_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in model_hash)
    ):
        errors.append("model_sha256 must be a 64-character lowercase SHA-256 digest.")

    metrics = metadata.get("test_metrics")
    if not isinstance(metrics, dict):
        errors.append("test_metrics is missing or is not an object.")
    else:
        for name in _REQUIRED_METRICS:
            if not _finite_number(metrics.get(name)):
                errors.append(f"test_metrics.{name} must be a finite number.")
        for name in _BOUNDED_METRICS:
            value = metrics.get(name)
            if _finite_number(value) and not 0 <= float(value) <= 1:
                errors.append(f"test_metrics.{name} must be between 0 and 1.")
        for name in ("log_loss", "pr_auc_lift_over_recency",
                     "lift_at_5pct", "lift_at_10pct", "lift_at_20pct"):
            value = metrics.get(name)
            if _finite_number(value) and float(value) < 0:
                errors.append(f"test_metrics.{name} must be non-negative.")
        mcc = metrics.get("matthews_corrcoef")
        if _finite_number(mcc) and not -1 <= float(mcc) <= 1:
            errors.append("test_metrics.matthews_corrcoef must be between -1 and 1.")
        gini = metrics.get("gini")
        if _finite_number(gini) and not -1 <= float(gini) <= 1:
            errors.append("test_metrics.gini must be between -1 and 1.")
    baseline = metadata.get("dummy_baseline")
    if not isinstance(baseline, dict):
        errors.append("dummy_baseline is missing or is not an object.")
    else:
        for name in ("train_positive_prevalence", "test_pr_auc",
                     "test_positive_prevalence", "test_brier_score", "test_log_loss",
                     "recency_only_pr_auc"):
            if not _finite_number(baseline.get(name)):
                errors.append(f"dummy_baseline.{name} must be a finite number.")
        if isinstance(metrics, dict):
            comparisons = (
                ("dummy_baseline_pr_auc", "test_pr_auc", "Dummy PR-AUC mismatch."),
                ("positive_prevalence", "test_positive_prevalence", "Test prevalence mismatch."),
                ("recency_only_pr_auc", "recency_only_pr_auc", "Recency baseline mismatch."),
            )
            for metric_key, baseline_key, message in comparisons:
                left, right = metrics.get(metric_key), baseline.get(baseline_key)
                if _finite_number(left) and _finite_number(right):
                    if abs(float(left) - float(right)) > 1e-9:
                        errors.append(message)
            lift = metrics.get("pr_auc_lift_over_dummy")
            ap = metrics.get("pr_auc")
            baseline_ap = baseline.get("test_pr_auc")
            if all(_finite_number(x) for x in (lift, ap, baseline_ap)):
                if abs(float(lift) - (float(ap) - float(baseline_ap))) > 1e-9:
                    errors.append("PR-AUC lift over dummy is inconsistent.")
            recency_lift = metrics.get("pr_auc_lift_over_recency")
            recency_ap = baseline.get("recency_only_pr_auc")
            if all(_finite_number(x) for x in (recency_lift, ap, recency_ap)):
                expected_lift = float(ap) / float(recency_ap) if float(recency_ap) > 0 else 0.0
                if abs(float(recency_lift) - expected_lift) > 1e-9:
                    errors.append("PR-AUC lift over recency baseline is inconsistent.")

    distribution = metadata.get("class_distribution")
    if not isinstance(distribution, dict):
        errors.append("class_distribution is missing or is not an object.")
    else:
        for partition in ("train", "validation", "test"):
            counts = distribution.get(partition)
            if not isinstance(counts, dict):
                errors.append(f"class_distribution.{partition} is missing.")
                continue
            for label in ("0", "1"):
                count = counts.get(label)
                if not isinstance(count, int) or count <= 0:
                    errors.append(f"class_distribution.{partition}.{label} must be a positive integer.")

    population = metadata.get("population_definition")
    if not isinstance(population, str) or "first delivered order" not in population:
        errors.append("population_definition must describe customers after their first delivered order.")

    periods = metadata.get("periods")
    if not isinstance(periods, dict):
        errors.append("periods is missing or is not an object.")
    else:
        try:
            train_start = date.fromisoformat(str(periods["train_start"]))
            train_end = date.fromisoformat(str(periods["train_end"]))
            validation_start = date.fromisoformat(str(periods["validation_start"]))
            validation_end = date.fromisoformat(str(periods["validation_end"]))
            test_start = date.fromisoformat(str(periods["test_start"]))
            test_end = date.fromisoformat(str(periods["test_end"]))
            embargo = int(periods.get("label_horizon_embargo_days", 90))
            if embargo < 0:
                raise ValueError("Embargo cannot be negative.")
            if not train_start <= train_end < validation_start <= validation_end < test_start <= test_end:
                errors.append("Train, validation, and test periods are not strictly chronological.")
            if train_end + timedelta(days=embargo) > validation_start:
                errors.append("Training and validation label windows overlap.")
            if validation_end + timedelta(days=embargo) > test_start:
                errors.append("Validation and final-test label windows overlap.")
        except (KeyError, TypeError, ValueError):
            errors.append("periods must contain valid ISO dates and an embargo length.")

    bootstrap = metadata.get("customer_bootstrap_pr_auc_95_ci")
    if not isinstance(bootstrap, dict):
        errors.append("customer_bootstrap_pr_auc_95_ci is missing.")
    else:
        lower, upper, replicates = (
            bootstrap.get("lower_95"),
            bootstrap.get("upper_95"),
            bootstrap.get("replicates"),
        )
        if not (_finite_number(lower) and _finite_number(upper) and lower <= upper):
            errors.append("Customer-bootstrap PR-AUC interval is invalid.")
        elif not 0 <= float(lower) <= float(upper) <= 1:
            errors.append("Customer-bootstrap PR-AUC interval must be within [0,1].")
        if not isinstance(replicates, int) or replicates < 30:
            errors.append("Customer-bootstrap interval must contain at least 30 valid replicates.")

    capacity_bootstrap = metadata.get("customer_bootstrap_top10_95_ci")
    if not isinstance(capacity_bootstrap, dict):
        errors.append("customer_bootstrap_top10_95_ci is missing.")
    else:
        for metric_name in ("precision", "recall", "lift", "flagged_fraction"):
            interval = capacity_bootstrap.get(metric_name)
            if not isinstance(interval, dict):
                errors.append(f"customer_bootstrap_top10_95_ci.{metric_name} is missing.")
                continue
            lower, upper, replicates = (
                interval.get("lower_95"),
                interval.get("upper_95"),
                interval.get("replicates"),
            )
            if not (_finite_number(lower) and _finite_number(upper) and lower <= upper):
                errors.append(f"customer-bootstrap interval for {metric_name} is invalid.")
            if not isinstance(replicates, int) or replicates < 30:
                errors.append(f"Customer-bootstrap interval for {metric_name} has too few replicates.")
            if metric_name != "lift":
                if _finite_number(lower) and not 0 <= float(lower) <= 1:
                    errors.append(f"Customer-bootstrap lower bound for {metric_name} must be in [0,1].")
                if _finite_number(upper) and not 0 <= float(upper) <= 1:
                    errors.append(f"Customer-bootstrap upper bound for {metric_name} must be in [0,1].")
            elif _finite_number(lower) and float(lower) < 0:
                errors.append("Customer-bootstrap lift lower bound must be non-negative.")

    unique_customers = metadata.get("unique_customers")
    if not isinstance(unique_customers, dict) or not isinstance(unique_customers.get("all"), int):
        errors.append("unique_customers metadata is missing.")
    elif unique_customers["all"] <= 0:
        errors.append("unique_customers.all must be positive.")

    return errors
