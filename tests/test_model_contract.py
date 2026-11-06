from copy import deepcopy

from src.features.churn_features import FEATURE_COLS
from src.models.model_contract import MODEL_VERSION, validate_model_metadata


def valid_metadata():
    return {
        "pipeline_status": "ready",
        "promotion_gate": {"production_eligible": True},
        "model_version": MODEL_VERSION,
        "feature_columns": FEATURE_COLS,
        "cohort_definition": "Eligible after at least one delivered order; predict the second delivered purchase within the next 90 days.",
        "evaluation_positive_class": "second_purchase_within_90_days (SQL churned=0)",
        "threshold_selection": "maximum F2 for second_purchase_within_90_days on separate validation period",
        "calibration": {
            "method": "Platt sigmoid",
            "fit_source": "purged out-of-fold predictions from training partition only",
            "slope": 0.8,
            "intercept": -1.2,
        },
        "threshold_beta": 2.0,
        "repeat_purchase_probability_threshold": 0.03,
        "feature_availability": {
            "review_score": "review_answer_timestamp < snapshot_date",
            "first_order_profile": "first delivered order purchase and delivery timestamps < snapshot_date; review score requires review_answer_timestamp < snapshot_date",
            "rolling_order_profile": "purchase_timestamp < snapshot_date and delivery_timestamp < snapshot_date within the trailing 180-day feature window",
            "lifetime_history": "purchase_timestamp < snapshot_date and delivered_timestamp < snapshot_date",
            "prior_order_window": "[snapshot_date - 360 days, snapshot_date - 180 days)"
        },
        "model_sha256": "a" * 64,
        "threshold": 0.97,
        "test_metrics": {
            "roc_auc": 0.72,
            "pr_auc": 0.63,
            "precision": 0.40,
            "recall": 0.45,
            "f1": 0.42,
            "f2": 0.55,
            "class_0_precision": 0.40,
            "class_0_recall": 0.45,
            "class_0_f1": 0.42,
            "class_1_precision": 0.95,
            "class_1_recall": 0.85,
            "class_1_f1": 0.90,
            "balanced_accuracy": 0.70,
            "matthews_corrcoef": 0.35,
            "positive_prevalence": 0.36363636363636365,
            "threshold": 0.97,
            "dummy_baseline_pr_auc": 0.36363636363636365,
            "pr_auc_lift_over_dummy": 0.26636363636363636,
            "brier_score": 0.10,
            "log_loss": 0.42,
            "expected_calibration_error": 0.03,
            "recency_baseline_pr_auc": 0.52,
            "frequency_baseline_pr_auc": 0.55,
            "lifetime_frequency_baseline_pr_auc": 0.57,
            "prior_dummy_brier_score": 0.22,
            "prior_dummy_log_loss": 0.61,
            "mean_predicted_repeat_probability": 0.34,
            "predicted_to_observed_rate_ratio": 0.93,
        },
        "test_capacity_metrics": {
            "5%": {"precision": 0.60, "recall": 0.10, "lift_over_monthly_prevalence": 1.6},
            "10%": {"precision": 0.50, "recall": 0.20, "lift_over_monthly_prevalence": 1.3},
            "20%": {"precision": 0.40, "recall": 0.40, "lift_over_monthly_prevalence": 1.1},
        },
        "baseline_capacity_metrics": {
            baseline: {
                "5%": {"precision": 0.50, "recall": 0.10, "lift_over_monthly_prevalence": 1.5},
                "10%": {"precision": 0.45, "recall": 0.20, "lift_over_monthly_prevalence": 1.3},
                "20%": {"precision": 0.35, "recall": 0.30, "lift_over_monthly_prevalence": 1.1},
            }
            for baseline in ("recency_only", "orders_180d_only", "lifetime_frequency_only")
        },
        "customer_cluster_bootstrap_pr_auc_ci95": {
            "lower_95": 0.58, "upper_95": 0.68, "valid_replicates": 200
        },
        "known_customer_test_metrics": {"rows": 50, "customers": 30},
        "unseen_customer_test_metrics": {"rows": 100, "customers": 80},
        "known_customer_test_metrics": {"rows": 50, "customers": 30},
        "dummy_baseline": {
            "strategy": "prior",
            "test_pr_auc": 0.36363636363636365,
            "test_positive_prevalence": 0.36363636363636365,
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



def test_evaluation_positive_class_must_be_minority_repeat_purchase():
    metadata = valid_metadata()
    metadata["evaluation_positive_class"] = "churned"
    assert any("evaluation_positive_class" in error for error in validate_model_metadata(metadata))


def test_repeat_purchase_threshold_is_required_and_bounded():
    metadata = valid_metadata()
    metadata["repeat_purchase_probability_threshold"] = 1.5
    assert any("repeat_purchase_probability_threshold" in error for error in validate_model_metadata(metadata))



def test_churn_and_repeat_probability_thresholds_must_be_complements():
    metadata = valid_metadata()
    metadata["threshold"] = 0.90
    metadata["test_metrics"]["threshold"] = 0.90
    errors = validate_model_metadata(metadata)
    assert any("must be complementary" in error for error in errors)



def test_model_contract_requires_f2_and_recall_oriented_threshold_policy():
    metadata = valid_metadata()
    metadata["test_metrics"].pop("f2")
    metadata["threshold_selection"] = "maximum F1"
    errors = validate_model_metadata(metadata)
    assert any("test_metrics.f2" in error for error in errors)
    assert any("threshold_selection" in error for error in errors)


def test_model_contract_rejects_future_looking_lifetime_feature_definition():
    metadata = valid_metadata()
    metadata["feature_availability"]["lifetime_history"] = (
        "all historical and future orders"
    )
    assert any("lifetime history" in error for error in validate_model_metadata(metadata))



def test_model_contract_requires_baseline_capacity_comparisons():
    metadata = valid_metadata()
    metadata.pop("baseline_capacity_metrics")
    errors = validate_model_metadata(metadata)
    assert any("baseline_capacity_metrics" in error for error in errors)



def test_model_contract_rejects_first_order_profile_leakage():
    metadata = valid_metadata()
    metadata["feature_availability"]["first_order_profile"] = (
        "all customer orders including future orders"
    )
    assert any("first_order_profile" in error for error in validate_model_metadata(metadata))



def test_model_contract_requires_known_customer_test_subset():
    metadata = valid_metadata()
    metadata.pop("known_customer_test_metrics")
    assert any("known_customer_test_metrics" in error for error in validate_model_metadata(metadata))



def test_research_only_status_is_valid_when_promotion_gate_fails():
    metadata = valid_metadata()
    metadata["pipeline_status"] = "research_only"
    metadata["promotion_gate"] = {"production_eligible": False}
    assert validate_model_metadata(metadata) == []


def test_pipeline_status_must_match_promotion_gate():
    metadata = valid_metadata()
    metadata["promotion_gate"]["production_eligible"] = False
    errors = validate_model_metadata(metadata)
    assert any("must agree with promotion_gate" in error for error in errors)
