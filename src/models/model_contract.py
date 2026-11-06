from __future__ import annotations

from datetime import date, timedelta
from math import isfinite
from typing import Any

from src.features.churn_features import FEATURE_COLS

MODEL_VERSION = "churn-pit-v5"

_REQUIRED_METRICS = (
    "roc_auc",
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
    "threshold",
    "dummy_baseline_pr_auc",
    "pr_auc_lift_over_dummy",
    "brier_score",
    "log_loss",
    "expected_calibration_error",
    "recency_baseline_pr_auc",
    "frequency_baseline_pr_auc",
    "lifetime_frequency_baseline_pr_auc",
    "prior_dummy_brier_score",
    "prior_dummy_log_loss",
    "mean_predicted_repeat_probability",
    "predicted_to_observed_rate_ratio",
)


def validate_model_metadata(metadata: dict[str, Any]) -> list[str]:
    """Return deployment-contract errors; an empty list means metadata is valid."""
    if not isinstance(metadata, dict):
        return ["Model metadata must be a JSON object."]

    errors: list[str] = []
    if metadata.get("pipeline_status") not in {"ready", "research_only"}:
        errors.append("pipeline_status must be 'ready' or 'research_only'.")
    if metadata.get("model_version") != MODEL_VERSION:
        errors.append(
            f"model_version must be {MODEL_VERSION!r}; found "
            f"{metadata.get('model_version')!r}."
        )
    gate = metadata.get("promotion_gate")
    if not isinstance(gate, dict) or not isinstance(gate.get("production_eligible"), bool):
        errors.append("promotion_gate must include a boolean production_eligible value.")
    elif (metadata.get("pipeline_status") == "ready") != gate["production_eligible"]:
        errors.append("pipeline_status must agree with promotion_gate.production_eligible.")

    if metadata.get("feature_columns") != FEATURE_COLS:
        errors.append("feature_columns do not match the application's feature contract.")

    expected_cohort = (
        "Eligible after at least one delivered order; predict the second delivered "
        "purchase within the next 90 days."
    )
    if metadata.get("cohort_definition") != expected_cohort:
        errors.append("cohort_definition must describe first-order customers and second-purchase prediction.")

    if metadata.get("evaluation_positive_class") != "second_purchase_within_90_days (SQL churned=0)":
        errors.append("evaluation_positive_class must identify second purchase within 90 days.")

    if metadata.get("threshold_selection") != "maximum F2 for second_purchase_within_90_days on separate validation period":
        errors.append("threshold_selection must use validation F2 for second-purchase propensity.")

    calibration = metadata.get("calibration")
    if not isinstance(calibration, dict):
        errors.append("calibration metadata is missing.")
    else:
        if calibration.get("method") != "Platt sigmoid":
            errors.append("calibration.method must identify Platt sigmoid calibration.")
        if calibration.get("fit_source") != "purged out-of-fold predictions from training partition only":
            errors.append("calibration must be fit on purged out-of-fold training predictions.")
        for name in ("slope", "intercept"):
            value = calibration.get(name)
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                errors.append(f"calibration.{name} must be finite.")
    threshold_beta = metadata.get("threshold_beta")
    if not isinstance(threshold_beta, (int, float)) or float(threshold_beta) != 2.0:
        errors.append("threshold_beta must be 2.0 for recall-oriented F2 selection.")

    threshold = metadata.get("threshold")
    repeat_threshold = metadata.get("repeat_purchase_probability_threshold")
    if not isinstance(repeat_threshold, (int, float)) or not isfinite(float(repeat_threshold)):
        errors.append("repeat_purchase_probability_threshold must be finite.")
    elif not 0 <= float(repeat_threshold) <= 1:
        errors.append("repeat_purchase_probability_threshold must be between 0 and 1.")
    elif isinstance(threshold, (int, float)) and isfinite(float(threshold)):
        if abs(float(threshold) + float(repeat_threshold) - 1.0) > 1e-12:
            errors.append(
                "Churn and repeat-purchase probability thresholds must be complementary."
            )

    availability = metadata.get("feature_availability")
    if not isinstance(availability, dict):
        errors.append("feature_availability is missing or is not an object.")
    elif availability.get("review_score") != "review_answer_timestamp < snapshot_date":
        errors.append("review_score must use review_answer_timestamp as its availability time.")
    else:
        if availability.get("first_order_profile") != (
            "first delivered order purchase and delivery timestamps < snapshot_date; "
            "review score requires review_answer_timestamp < snapshot_date"
        ):
            errors.append("first_order_profile must use only pre-snapshot order and review information.")
        if availability.get("rolling_order_profile") != (
            "purchase_timestamp < snapshot_date and delivery_timestamp < snapshot_date "
            "within the trailing 180-day feature window"
        ):
            errors.append("rolling_order_profile must use only delivered pre-snapshot orders.")
        if availability.get("lifetime_history") != (
            "purchase_timestamp < snapshot_date and delivered_timestamp < snapshot_date"
        ):
            errors.append("lifetime history must be limited to orders observable before the snapshot.")
        if availability.get("prior_order_window") != (
            "[snapshot_date - 360 days, snapshot_date - 180 days)"
        ):
            errors.append("prior_order_window must end before the current 180-day window.")

    model_hash = metadata.get("model_sha256")
    if (
        not isinstance(model_hash, str)
        or len(model_hash) != 64
        or any(character not in "0123456789abcdef" for character in model_hash)
    ):
        errors.append("model_sha256 must be a 64-character lowercase SHA-256 digest.")

    if not isinstance(threshold, (int, float)) or not isfinite(float(threshold)):
        errors.append("threshold must be a finite number.")
    elif not 0 <= float(threshold) <= 1:
        errors.append("threshold must be between 0 and 1.")

    metrics = metadata.get("test_metrics")
    if not isinstance(metrics, dict):
        errors.append("test_metrics is missing or is not an object.")
    else:
        for name in _REQUIRED_METRICS:
            value = metrics.get(name)
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                errors.append(f"test_metrics.{name} must be a finite number.")
        bounded_metrics = (
            "roc_auc", "pr_auc", "precision", "recall", "f1", "f2",
            "class_0_precision", "class_0_recall", "class_0_f1",
            "class_1_precision", "class_1_recall", "class_1_f1",
            "balanced_accuracy", "positive_prevalence", "threshold",
            "dummy_baseline_pr_auc", "brier_score", "prior_dummy_brier_score",
            "expected_calibration_error", "mean_predicted_repeat_probability",
            "recency_baseline_pr_auc", "frequency_baseline_pr_auc",
            "lifetime_frequency_baseline_pr_auc",
        )
        for name in bounded_metrics:
            value = metrics.get(name)
            if isinstance(value, (int, float)) and isfinite(float(value)):
                if not 0 <= float(value) <= 1:
                    errors.append(f"test_metrics.{name} must be between 0 and 1.")
        for name in ("log_loss", "prior_dummy_log_loss"):
            value = metrics.get(name)
            if isinstance(value, (int, float)) and isfinite(float(value)) and float(value) < 0:
                errors.append(f"test_metrics.{name} cannot be negative.")
        ratio = metrics.get("predicted_to_observed_rate_ratio")
        if isinstance(ratio, (int, float)) and isfinite(float(ratio)) and float(ratio) < 0:
            errors.append("test_metrics.predicted_to_observed_rate_ratio cannot be negative.")
        metric_threshold = metrics.get("threshold")
        mcc = metrics.get("matthews_corrcoef")
        if isinstance(mcc, (int, float)) and isfinite(float(mcc)):
            if not -1 <= float(mcc) <= 1:
                errors.append("test_metrics.matthews_corrcoef must be between -1 and 1.")
        if (
            isinstance(metric_threshold, (int, float))
            and isinstance(threshold, (int, float))
            and abs(float(metric_threshold) - float(threshold)) > 1e-12
        ):
            errors.append("test_metrics.threshold does not match metadata.threshold.")

    baseline = metadata.get("dummy_baseline")
    if not isinstance(baseline, dict):
        errors.append("dummy_baseline is missing or is not an object.")
    else:
        for name in ("test_pr_auc", "test_positive_prevalence"):
            value = baseline.get(name)
            if not isinstance(value, (int, float)) or not isfinite(float(value)):
                errors.append(f"dummy_baseline.{name} must be a finite number.")
            elif not 0 <= float(value) <= 1:
                errors.append(f"dummy_baseline.{name} must be between 0 and 1.")
        if isinstance(metrics, dict):
            metric_baseline = metrics.get("dummy_baseline_pr_auc")
            baseline_ap = baseline.get("test_pr_auc")
            if (
                isinstance(metric_baseline, (int, float))
                and isinstance(baseline_ap, (int, float))
                and abs(float(metric_baseline) - float(baseline_ap)) > 1e-9
            ):
                errors.append("Dummy PR-AUC differs between test_metrics and dummy_baseline.")
            metric_prevalence = metrics.get("positive_prevalence")
            baseline_prevalence = baseline.get("test_positive_prevalence")
            if (
                isinstance(metric_prevalence, (int, float))
                and isinstance(baseline_prevalence, (int, float))
                and abs(float(metric_prevalence) - float(baseline_prevalence)) > 1e-9
            ):
                errors.append("Test positive prevalence differs from the dummy baseline prevalence.")
            lift = metrics.get("pr_auc_lift_over_dummy")
            pr_auc = metrics.get("pr_auc")
            if all(isinstance(value, (int, float)) for value in (lift, pr_auc, baseline_ap)):
                if abs(float(lift) - (float(pr_auc) - float(baseline_ap))) > 1e-9:
                    errors.append("PR-AUC lift is inconsistent with the model and dummy scores.")

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
                    errors.append(
                        f"class_distribution.{partition}.{label} must be a positive integer."
                    )

    capacity_metrics = metadata.get("test_capacity_metrics")
    if not isinstance(capacity_metrics, dict):
        errors.append("test_capacity_metrics is missing or is not an object.")
    else:
        for capacity in ("5%", "10%", "20%"):
            values = capacity_metrics.get(capacity)
            if not isinstance(values, dict):
                errors.append(f"test_capacity_metrics.{capacity} is missing.")
                continue
            for metric_name in ("precision", "recall", "lift_over_monthly_prevalence"):
                value = values.get(metric_name)
                if not isinstance(value, (int, float)) or not isfinite(float(value)):
                    errors.append(
                        f"test_capacity_metrics.{capacity}.{metric_name} must be finite."
                    )
                elif metric_name != "lift_over_monthly_prevalence" and not 0 <= float(value) <= 1:
                    errors.append(
                        f"test_capacity_metrics.{capacity}.{metric_name} must be between 0 and 1."
                    )

    baseline_capacity = metadata.get("baseline_capacity_metrics")
    if not isinstance(baseline_capacity, dict):
        errors.append("baseline_capacity_metrics is missing or is not an object.")
    else:
        for baseline_name in ("recency_only", "orders_180d_only", "lifetime_frequency_only"):
            baseline_values = baseline_capacity.get(baseline_name)
            if not isinstance(baseline_values, dict):
                errors.append(f"baseline_capacity_metrics.{baseline_name} is missing.")
                continue
            for capacity in ("5%", "10%", "20%"):
                values = baseline_values.get(capacity)
                if not isinstance(values, dict):
                    errors.append(
                        f"baseline_capacity_metrics.{baseline_name}.{capacity} is missing."
                    )
                    continue
                for metric_name in ("precision", "recall", "lift_over_monthly_prevalence"):
                    value = values.get(metric_name)
                    if not isinstance(value, (int, float)) or not isfinite(float(value)):
                        errors.append(
                            f"baseline_capacity_metrics.{baseline_name}.{capacity}.{metric_name} must be finite."
                        )
                    elif metric_name != "lift_over_monthly_prevalence" and not 0 <= float(value) <= 1:
                        errors.append(
                            f"baseline_capacity_metrics.{baseline_name}.{capacity}.{metric_name} must be between 0 and 1."
                        )

    ci = metadata.get("customer_cluster_bootstrap_pr_auc_ci95")
    if not isinstance(ci, dict):
        errors.append("customer_cluster_bootstrap_pr_auc_ci95 is missing.")
    else:
        lower, upper = ci.get("lower_95"), ci.get("upper_95")
        if not all(isinstance(v, (int, float)) and isfinite(float(v)) for v in (lower, upper)):
            errors.append("Customer-clustered PR-AUC confidence interval must be finite.")
        elif not 0 <= float(lower) <= float(upper) <= 1:
            errors.append("Customer-clustered PR-AUC confidence interval is invalid.")
        replicates = ci.get("valid_replicates")
        if not isinstance(replicates, int) or replicates <= 0:
            errors.append("Customer-clustered PR-AUC interval must have valid bootstrap replicates.")

    known = metadata.get("known_customer_test_metrics")
    if not isinstance(known, dict):
        errors.append("known_customer_test_metrics is missing.")
    else:
        for name in ("rows", "customers"):
            value = known.get(name)
            if not isinstance(value, int) or value < 0:
                errors.append(f"known_customer_test_metrics.{name} must be a non-negative integer.")

    unseen = metadata.get("unseen_customer_test_metrics")
    if not isinstance(unseen, dict):
        errors.append("unseen_customer_test_metrics is missing.")
    else:
        for name in ("rows", "customers"):
            value = unseen.get(name)
            if not isinstance(value, int) or value < 0:
                errors.append(f"unseen_customer_test_metrics.{name} must be a non-negative integer.")

    periods = metadata.get("periods")
    if not isinstance(periods, dict):
        errors.append("periods is missing or is not an object.")
    else:
        try:
            train_end = date.fromisoformat(str(periods["train_end"]))
            validation_start = date.fromisoformat(str(periods["validation_start"]))
            validation_end = date.fromisoformat(str(periods["validation_end"]))
            test_start = date.fromisoformat(str(periods["test_start"]))
            embargo = int(periods.get("label_horizon_embargo_days", 90))
            if embargo < 0:
                raise ValueError("Embargo cannot be negative.")
            train_start = date.fromisoformat(str(periods["train_start"]))
            validation_end = date.fromisoformat(str(periods["validation_end"]))
            test_end = date.fromisoformat(str(periods["test_end"]))
            if not train_start <= train_end < validation_start <= validation_end < test_start <= test_end:
                errors.append("Train, validation, and test periods are not strictly chronological.")
            if train_end + timedelta(days=embargo) > validation_start:
                errors.append("Training and validation label windows overlap.")
            if validation_end + timedelta(days=embargo) > test_start:
                errors.append("Validation and final-test label windows overlap.")
        except (KeyError, TypeError, ValueError):
            errors.append("periods must contain valid ISO dates and an embargo length.")

    return errors
