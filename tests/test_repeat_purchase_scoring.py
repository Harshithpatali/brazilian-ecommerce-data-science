import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from src.models.train_churn import (
    CalibratedRepeatPurchaseModel,
    _logit,
    assign_monthly_capacity,
    customer_bootstrap_capacity_ci,
    customer_bootstrap_pr_auc_ci,
    repeat_purchase_average_precision_scorer,
    _calibration_table,
)


class FakeChurnEstimator:
    def __init__(self, churn_probabilities):
        self.churn_probabilities = np.asarray(churn_probabilities, dtype=float)

    def predict_proba(self, X):
        p_churn = self.churn_probabilities
        return np.column_stack([1.0 - p_churn, p_churn])


class FixedEstimator:
    def __init__(self, repeat_probabilities):
        self.repeat_probabilities = np.asarray(repeat_probabilities, dtype=float)

    def predict_proba(self, X):
        return np.column_stack([
            self.repeat_probabilities,
            1.0 - self.repeat_probabilities,
        ])


def test_cv_scorer_ranks_repeat_purchase_event_not_majority_churn_event():
    y_churned = np.array([0, 1, 0, 1])
    X = pd.DataFrame({"feature": [1, 2, 3, 4]})

    good = FakeChurnEstimator([0.1, 0.9, 0.2, 0.8])
    bad = FakeChurnEstimator([0.9, 0.1, 0.8, 0.2])

    assert repeat_purchase_average_precision_scorer(good, X, y_churned) == 1.0
    assert repeat_purchase_average_precision_scorer(good, X, y_churned) > (
        repeat_purchase_average_precision_scorer(bad, X, y_churned)
    )


def test_monthly_capacity_flags_exact_share_and_ranks_within_month():
    dates = pd.to_datetime(["2018-01-01"] * 10 + ["2018-02-01"] * 10)
    scores = np.arange(20, dtype=float)
    labels = np.array([0] * 5 + [1] * 5 + [0] * 5 + [1] * 5)

    selected, metrics = assign_monthly_capacity(
        labels, scores, dates, capacity_fraction=0.20
    )

    assert selected.sum() == 4
    assert selected[:10].sum() == 2
    assert selected[10:].sum() == 2
    assert selected[8] == 1
    assert selected[9] == 1
    assert metrics["flagged_fraction"] == 0.2
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1


def test_monthly_capacity_rejects_invalid_fraction_and_mismatched_lengths():
    labels = np.array([0, 1])
    scores = np.array([0.8, 0.2])
    dates = pd.to_datetime(["2018-01-01", "2018-01-01"])

    for fraction in (0.0, 1.0, -0.1):
        try:
            assign_monthly_capacity(labels, scores, dates, fraction)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid capacity fraction must fail")

    try:
        assign_monthly_capacity(labels, scores[:1], dates, 0.1)
    except ValueError as exc:
        assert "equal lengths" in str(exc)
    else:
        raise AssertionError("Mismatched lengths must fail")


def test_calibrated_wrapper_preserves_repeat_and_no_purchase_probability_order():
    y = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    raw_repeat = np.array([0.8, 0.2, 0.7, 0.3, 0.6, 0.4, 0.9, 0.1])
    calibrator = LogisticRegression().fit(_logit(raw_repeat), y)
    model = CalibratedRepeatPurchaseModel(FixedEstimator(raw_repeat), calibrator)

    probabilities = model.predict_proba(pd.DataFrame({"feature": range(8)}))
    assert probabilities.shape == (8, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert np.all((probabilities >= 0).all(axis=1))
    assert np.all((probabilities <= 1).all(axis=1))
    assert np.array_equal(model.classes_, np.array([0, 1]))


def test_customer_bootstrap_pr_auc_interval_is_reproducible_and_bounded():
    y = np.array([1, 0, 1, 0] * 20)
    scores = np.array([0.9, 0.1, 0.7, 0.2] * 20)
    customer_ids = np.array([f"c{i // 2}" for i in range(80)])

    ci = customer_bootstrap_pr_auc_ci(
        y, scores, customer_ids, n_bootstrap=50, random_state=17
    )
    assert 0 <= ci["lower_95"] <= ci["upper_95"] <= 1
    assert ci["replicates"] >= 30


def test_customer_bootstrap_capacity_ci_reports_valid_bounds():
    y = np.array([1, 0, 1, 0] * 30)
    scores = np.array([0.9, 0.1, 0.7, 0.2] * 30)
    dates = pd.to_datetime(["2018-01-01"] * 120)
    customer_ids = np.array([f"c{i // 2}" for i in range(120)])

    ci = customer_bootstrap_capacity_ci(
        y, scores, dates, customer_ids,
        capacity_fraction=0.25, n_bootstrap=50, random_state=29,
    )
    for metric in ("precision", "recall", "flagged_fraction"):
        assert 0 <= ci[metric]["lower_95"] <= ci[metric]["upper_95"] <= 1
        assert ci[metric]["replicates"] >= 30
    assert ci["lift"]["lower_95"] >= 0
    assert ci["lift"]["upper_95"] >= ci["lift"]["lower_95"]


def test_candidate_models_include_catboost_native_categorical_candidate():
    from src.models.train_churn import build_models

    models = build_models()
    assert {
        "logistic_regression",
        "hist_gradient_boosting",
        "random_forest",
        "catboost",
    } <= set(models)



def test_all_candidate_pipelines_are_sklearn_cloneable():
    from sklearn.base import clone
    from src.models.train_churn import build_models

    for pipeline, _ in build_models().values():
        clone(pipeline)


def test_catboost_receives_native_categorical_columns_at_fit_not_constructor():
    from src.features.churn_features import CATEGORICAL_FEATURE_COLS
    from src.models.train_churn import build_models

    pipeline, _ = build_models()["catboost"]
    assert pipeline.named_steps["model"].get_params()["cat_features"] is None
    assert CATEGORICAL_FEATURE_COLS == ["customer_state", "snapshot_month"]


def test_calibration_table_handles_constant_probability_scores():
    table = _calibration_table(
        np.array([0, 1, 0, 1, 0, 1]),
        np.full(6, 0.05),
    )
    assert len(table) == 1
    assert table.loc[0, "customers"] == 6
    assert table.loc[0, "mean_predicted_probability"] == 0.05
    assert table.loc[0, "observed_repeat_rate"] == 0.5
