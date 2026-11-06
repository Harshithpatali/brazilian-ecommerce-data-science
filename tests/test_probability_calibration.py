import numpy as np
import pandas as pd

from src.models.calibration import (
    PlattCalibratedChurnClassifier,
    fit_platt_calibrator,
    probability_logit,
)
from src.models.train_churn import (
    _cluster_bootstrap_pr_auc_ci,
    _monthly_capacity_metrics,
)


class FakeChurnEstimator:
    """A small fitted-style estimator whose class 1 probability is churn."""

    def __init__(self, repeat_probabilities):
        self.repeat_probabilities = np.asarray(repeat_probabilities, dtype=float)

    def predict_proba(self, X):
        repeat = self.repeat_probabilities[:len(X)]
        churn = 1.0 - repeat
        return np.column_stack([repeat, churn])


def test_probability_logit_clips_boundary_values_and_is_finite():
    logits = probability_logit(np.array([0.0, 0.2, 0.8, 1.0]))
    assert logits.shape == (4, 1)
    assert np.isfinite(logits).all()


def test_platt_calibrator_outputs_repeat_then_churn_probability_columns():
    raw_repeat = np.array([0.01, 0.02, 0.05, 0.10, 0.3, 0.5, 0.7, 0.9])
    actual_repeat = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    calibrator = fit_platt_calibrator(raw_repeat, actual_repeat)
    model = PlattCalibratedChurnClassifier(
        FakeChurnEstimator(raw_repeat), calibrator
    )

    probabilities = model.predict_proba(pd.DataFrame({"x": np.arange(8)}))

    assert probabilities.shape == (8, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))
    assert probabilities[-1, 0] > probabilities[0, 0]
    assert np.allclose(probabilities[:, 1], 1.0 - probabilities[:, 0])


def test_monthly_capacity_metrics_rank_within_each_snapshot_cohort():
    dates = pd.to_datetime(
        ["2018-01-01"] * 10 + ["2018-02-01"] * 10
    )
    y = np.array([1] + [0] * 9 + [1] + [0] * 9)
    scores = np.array(
        [0.99] + list(np.linspace(0.1, 0.01, 9))
        + [0.98] + list(np.linspace(0.2, 0.02, 9))
    )

    result = _monthly_capacity_metrics(y, scores, dates, 0.10)

    assert result["customers_contacted"] == 2
    assert result["true_repeat_purchases_captured"] == 2
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0
    assert result["lift_over_monthly_prevalence"] == 10.0


def test_customer_cluster_bootstrap_returns_ordered_finite_interval():
    y = np.array([1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0])
    scores = np.array([0.9, 0.8, 0.2, 0.7, 0.6, 0.1,
                       0.95, 0.5, 0.4, 0.85, 0.3, 0.05])
    customers = np.array(["a", "a", "b", "c", "c", "d",
                          "e", "e", "f", "g", "g", "h"])

    interval = _cluster_bootstrap_pr_auc_ci(
        y, scores, customers, n_bootstrap=30, random_state=7
    )

    assert interval["valid_replicates"] > 0
    assert 0.0 <= interval["lower_95"] <= interval["upper_95"] <= 1.0



def test_equal_frequency_calibration_bins_have_balanced_support():
    from src.models.train_churn import _calibration_bins, _expected_calibration_error

    y = np.array([0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0])
    p = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006,
                  0.007, 0.008, 0.009, 0.010, 0.011, 0.012])
    bins = _calibration_bins(y, p, n_bins=4)

    assert len(bins) == 4
    assert bins["count"].max() - bins["count"].min() <= 1
    assert bins["count"].sum() == len(y)
    assert 0 <= _expected_calibration_error(y, p, n_bins=4) <= 1
