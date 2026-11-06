from copy import deepcopy

from src.features.churn_features import FEATURE_COLS
from src.models.model_contract import MODEL_VERSION, validate_model_metadata


def valid_metadata():
    return {
        "pipeline_status": "ready",
        "model_version": MODEL_VERSION,
        "feature_columns": FEATURE_COLS,
        "feature_availability": {
            "review_score": "review_answer_timestamp < snapshot_date"
        },
        "model_sha256": "a" * 64,
        "threshold": 0.44,
        "test_metrics": {
            "roc_auc": 0.72,
            "pr_auc": 0.63,
            "precision": 0.70,
            "recall": 0.60,
            "f1": 0.65,
            "positive_prevalence": 0.58,
            "threshold": 0.44,
            "dummy_baseline_pr_auc": 0.58,
            "pr_auc_lift_over_dummy": 0.05,
        },
        "dummy_baseline": {
            "strategy": "prior",
            "test_pr_auc": 0.58,
            "test_positive_prevalence": 0.58,
        },
        "class_distribution": {
            "train": {"0": 100, "1": 150},
            "validation": {"0": 30, "1": 40},
            "test": {"0": 20, "1": 35},
        },
        "periods": {
            "train_start": "2016-09-01",
            "train_end": "2017-01-01",
            "validation_start": "2017-04-01",
            "validation_end": "2017-06-01",
            "test_start": "2017-08-30",
            "test_end": "2017-12-01",
            "label_horizon_embargo_days": "90",
        },
    }


def test_valid_model_metadata_passes_contract():
    assert validate_model_metadata(valid_metadata()) == []


def test_stale_model_version_is_rejected():
    metadata = valid_metadata()
    metadata["model_version"] = "legacy"
    assert any("model_version" in error for error in validate_model_metadata(metadata))


def test_feature_order_or_schema_mismatch_is_rejected():
    metadata = valid_metadata()
    metadata["feature_columns"] = list(reversed(FEATURE_COLS))
    assert any("feature_columns" in error for error in validate_model_metadata(metadata))


def test_invalid_threshold_is_rejected():
    metadata = valid_metadata()
    metadata["threshold"] = 1.5
    metadata["test_metrics"]["threshold"] = 1.5
    assert any("threshold" in error for error in validate_model_metadata(metadata))


def test_missing_class_partition_is_rejected():
    metadata = valid_metadata()
    metadata["class_distribution"]["test"]["1"] = 0
    assert any("class_distribution.test.1" in error for error in validate_model_metadata(metadata))


def test_overlapping_label_windows_are_rejected():
    metadata = valid_metadata()
    metadata["periods"]["validation_start"] = "2017-03-01"
    errors = validate_model_metadata(metadata)
    assert any("Training and validation label windows overlap" in error for error in errors)


def test_missing_test_metrics_are_rejected():
    metadata = valid_metadata()
    metadata.pop("test_metrics")
    assert any("test_metrics" in error for error in validate_model_metadata(metadata))



def test_missing_model_hash_is_rejected():
    metadata = valid_metadata()
    metadata.pop("model_sha256")
    assert any("model_sha256" in error for error in validate_model_metadata(metadata))


def test_review_creation_timestamp_is_not_accepted_as_availability_time():
    metadata = valid_metadata()
    metadata["feature_availability"]["review_score"] = (
        "review_creation_date < snapshot_date"
    )
    assert any("review_answer_timestamp" in error for error in validate_model_metadata(metadata))


def test_dummy_baseline_and_lift_must_be_consistent():
    metadata = valid_metadata()
    metadata["test_metrics"]["pr_auc_lift_over_dummy"] = 0.10
    assert any("PR-AUC lift is inconsistent" in error for error in validate_model_metadata(metadata))
