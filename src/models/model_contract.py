from __future__ import annotations

from datetime import date, timedelta
from math import isfinite
from typing import Any

from src.features.churn_features import FEATURE_COLS

MODEL_VERSION = "churn-pit-v3"

_REQUIRED_METRICS = (
    "roc_auc",
    "pr_auc",
    "precision",
    "recall",
    "f1",
    "positive_prevalence",
    "threshold",
    "dummy_baseline_pr_auc",
    "pr_auc_lift_over_dummy",
)


def validate_model_metadata(metadata: dict[str, Any]) -> list[str]:
    """Return deployment-contract errors; an empty list means metadata is valid."""
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
        errors.append("feature_columns do not match the application's feature contract.")

    threshold = metadata.get("threshold")
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
            "roc_auc", "pr_auc", "precision", "recall", "f1",
            "positive_prevalence", "threshold", "dummy_baseline_pr_auc",
        )
        for name in bounded_metrics:
            value = metrics.get(name)
            if isinstance(value, (int, float)) and isfinite(float(value)):
                if not 0 <= float(value) <= 1:
                    errors.append(f"test_metrics.{name} must be between 0 and 1.")
        metric_threshold = metrics.get("threshold")
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
            if train_end + timedelta(days=embargo) > validation_start:
                errors.append("Training and validation label windows overlap.")
            if validation_end + timedelta(days=embargo) > test_start:
                errors.append("Validation and final-test label windows overlap.")
        except (KeyError, TypeError, ValueError):
            errors.append("periods must contain valid ISO dates and an embargo length.")

    return errors
