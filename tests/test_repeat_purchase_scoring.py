import numpy as np
import pandas as pd

from src.models.train_churn import (
    choose_fbeta_threshold,
    repeat_purchase_average_precision_scorer,
)


class FakeChurnEstimator:
    def __init__(self, churn_probabilities):
        self.churn_probabilities = np.asarray(churn_probabilities, dtype=float)

    def predict_proba(self, X):
        p_churn = self.churn_probabilities
        return np.column_stack([1.0 - p_churn, p_churn])


def test_cv_scorer_ranks_repeat_purchase_event_not_majority_churn_event():
    y_churned = np.array([0, 1, 0, 1])
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})

    good = FakeChurnEstimator([0.1, 0.9, 0.2, 0.8])
    bad = FakeChurnEstimator([0.9, 0.1, 0.8, 0.2])

    assert repeat_purchase_average_precision_scorer(good, X, y_churned) == 1.0
    assert repeat_purchase_average_precision_scorer(good, X, y_churned) > (
        repeat_purchase_average_precision_scorer(bad, X, y_churned)
    )


def test_f2_threshold_prefers_recall_at_least_as_much_as_f1():
    y_churned = np.array([0, 0, 0, 1, 1, 1])
    churn_probabilities = np.array([0.1, 0.4, 0.7, 0.8, 0.9, 0.95])

    f1_churn_cutoff, f1_metrics = choose_fbeta_threshold(
        y_churned, churn_probabilities, beta=1.0
    )
    f2_churn_cutoff, f2_metrics = choose_fbeta_threshold(
        y_churned, churn_probabilities, beta=2.0
    )

    assert f2_metrics["recall"] >= f1_metrics["recall"]
    assert f2_metrics["repeat_purchase_probability_threshold"] <= (
        f1_metrics["repeat_purchase_probability_threshold"]
    )
    assert abs(
        f2_churn_cutoff + f2_metrics["repeat_purchase_probability_threshold"] - 1.0
    ) < 1e-12
    assert f2_metrics["f2"] >= 0.0


def test_fbeta_rejects_non_positive_beta():
    y_churned = np.array([0, 1])
    probabilities = np.array([0.1, 0.9])
    try:
        choose_fbeta_threshold(y_churned, probabilities, beta=0)
    except ValueError as exc:
        assert "beta" in str(exc)
    else:
        raise AssertionError("Expected an invalid beta to raise ValueError")



def test_candidate_models_include_regularized_linear_boosted_and_forest_models():
    from src.models.train_churn import build_models

    models = build_models()
    assert {"logistic_regression", "hist_gradient_boosting", "random_forest"} <= set(models)



def test_model_search_includes_rfm_baselines_alongside_ml_estimators():
    from src.models.train_churn import build_models

    models = build_models()
    assert {
        "baseline_recency_only",
        "baseline_recent_frequency_only",
        "baseline_lifetime_frequency_only",
    } <= set(models)
