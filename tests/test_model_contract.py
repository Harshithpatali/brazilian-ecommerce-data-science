from src.features.churn_features import FEATURE_COLS
from src.models.model_contract import MODEL_VERSION, validate_model_metadata


def valid_metadata():
    metrics = {
        "roc_auc": 0.70,
        "pr_auc": 0.10,
        "precision": 0.10,
        "recall": 0.20,
        "f1": 0.133333,
        "f2": 0.166667,
        "class_0_precision": 0.10,
        "class_0_recall": 0.20,
        "class_0_f1": 0.133333,
        "class_1_precision": 0.97,
        "class_1_recall": 0.90,
        "class_1_f1": 0.934,
        "balanced_accuracy": 0.55,
        "matthews_corrcoef": 0.12,
        "positive_prevalence": 0.03,
        "dummy_baseline_pr_auc": 0.03,
        "pr_auc_lift_over_dummy": 0.07,
        "recency_only_pr_auc": 0.08,
        "pr_auc_lift_over_recency": 1.25,
        "brier_score": 0.028,
        "log_loss": 0.12,
        "brier_skill_score": -0.05,
        "precision_at_5pct": 0.12,
        "recall_at_5pct": 0.06,
        "lift_at_5pct": 4.0,
        "flagged_fraction_at_5pct": 0.05,
        "precision_at_10pct": 0.10,
        "recall_at_10pct": 0.20,
        "lift_at_10pct": 3.333333,
        "flagged_fraction_at_10pct": 0.10,
        "precision_at_20pct": 0.08,
        "recall_at_20pct": 0.32,
        "lift_at_20pct": 2.666667,
        "flagged_fraction_at_20pct": 0.20,
    }
    return {
        "pipeline_status": "ready",
        "model_version": MODEL_VERSION,
        "feature_columns": FEATURE_COLS,
        "evaluation_positive_class": "purchase_within_90_days (SQL churned=0)",
        "threshold_policy": "top_10_percent_per_snapshot_date",
        "capacity_fraction": 0.10,
        "threshold_selection": (
            "validation 90th percentile calibrated repeat-purchase probability; "
            "operational ranking selects top 10 percent per monthly snapshot"
        ),
        "calibration_method": "Platt sigmoid fitted on separate chronological validation period",
        "repeat_purchase_probability_threshold": 0.04,
        "threshold": 0.96,
        "feature_availability": {
            "review_score": "review_answer_timestamp < snapshot_date",
            "lifetime_history": "purchase_timestamp < snapshot_date and delivered_timestamp < snapshot_date",
            "prior_order_window": "[snapshot_date - 360 days, snapshot_date - 180 days)",
            "customer_state": "state from most recent delivered order observable before snapshot_date",
            "snapshot_month": "calendar month of snapshot_date",
            "future_orders": "used only to construct the target, never as predictors",
        },
        "model_sha256": "a" * 64,
        "test_metrics": metrics,
        "dummy_baseline": {
            "strategy": "training_prevalence",
            "train_positive_prevalence": 0.04,
            "test_pr_auc": 0.03,
            "test_positive_prevalence": 0.03,
            "test_brier_score": 0.029,
            "test_log_loss": 0.14,
            "recency_only_pr_auc": 0.08,
        },
        "class_distribution": {
            "all": {"0": 300, "1": 9700},
            "train": {"0": 150, "1": 4850},
            "validation": {"0": 75, "1": 2425},
            "test": {"0": 75, "1": 2425},
        },
        "population_definition": (
            "Monthly customer snapshots after each customer's first delivered order, "
            "when at least one delivered order was purchased and delivered in the prior 180 days."
        ),
        "periods": {
            "train_start": "2016-09-01",
            "train_end": "2017-01-01",
            "validation_start": "2017-04-01",
            "validation_end": "2017-06-01",
            "test_start": "2017-08-30",
            "test_end": "2017-12-01",
            "label_horizon_embargo_days": "90",
        },
        "customer_bootstrap_pr_auc_95_ci": {
            "lower_95": 0.08,
            "upper_95": 0.12,
            "replicates": 100,
        },
        "unique_customers": {
            "all": 300,
            "train": 150,
            "validation": 75,
            "test": 75,
        },
    }


def test_valid_v5_model_metadata_passes_contract():
    assert validate_model_metadata(valid_metadata()) == []


def test_stale_model_version_is_rejected():
    metadata = valid_metadata()
    metadata["model_version"] = "churn-pit-v4"
    assert any("model_version" in error for error in validate_model_metadata(metadata))


def test_feature_order_or_schema_mismatch_is_rejected():
    metadata = valid_metadata()
    metadata["feature_columns"] = list(reversed(FEATURE_COLS))
    assert any("feature_columns" in error for error in validate_model_metadata(metadata))


def test_invalid_reference_threshold_is_rejected():
    metadata = valid_metadata()
    metadata["threshold"] = 1.5
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
    metadata["feature_availability"]["review_score"] = "review_creation_date < snapshot_date"
    assert any("review_score" in error for error in validate_model_metadata(metadata))


def test_model_contract_requires_top_decile_and_calibration_policy():
    metadata = valid_metadata()
    metadata["threshold_policy"] = "maximum F2 threshold"
    metadata["calibration_method"] = "calibrated on final test"
    errors = validate_model_metadata(metadata)
    assert any("threshold_policy" in error for error in errors)
    assert any("calibration_method" in error for error in errors)


def test_model_contract_rejects_future_looking_state_and_history_features():
    metadata = valid_metadata()
    metadata["feature_availability"]["customer_state"] = "latest state"
    metadata["feature_availability"]["lifetime_history"] = "all orders"
    errors = validate_model_metadata(metadata)
    assert any("customer_state" in error for error in errors)
    assert any("lifetime_history" in error for error in errors)


def test_negative_brier_skill_is_a_valid_evaluation_result():
    metadata = valid_metadata()
    metadata["test_metrics"]["brier_skill_score"] = -0.25
    assert validate_model_metadata(metadata) == []


def test_customer_bootstrap_interval_must_be_valid():
    metadata = valid_metadata()
    metadata["customer_bootstrap_pr_auc_95_ci"]["lower_95"] = 0.20
    assert any("customer-bootstrap PR-AUC interval" in error for error in validate_model_metadata(metadata))



def test_capacity_policy_holdout_metrics_do_not_claim_one_global_test_threshold():
    assert "threshold" not in valid_metadata()["test_metrics"]
