import numpy as np
import pandas as pd

from src.models.train_churn import (
    choose_f1_threshold,
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


def test_validation_f1_threshold_targets_repeat_purchase_and_returns_churn_cutoff():
    y_churned = np.array([0, 1, 0, 1])
    p_churn = np.array([0.1, 0.9, 0.2, 0.8])

    churn_cutoff, metrics = choose_f1_threshold(y_churned, p_churn)

    assert 0.19 < churn_cutoff < 0.21
    assert metrics["f1"] == 1.0
    assert abs(
        churn_cutoff + metrics["repeat_purchase_probability_threshold"] - 1.0
    ) < 1e-12
