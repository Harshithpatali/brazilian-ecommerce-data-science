from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def probability_logit(probabilities: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """Convert probabilities to finite logits with safe clipping."""
    values = np.asarray(probabilities, dtype=float)
    if values.ndim != 1:
        raise ValueError("probabilities must be a one-dimensional array.")
    if not np.isfinite(values).all():
        raise ValueError("probabilities must contain only finite values.")
    values = np.clip(values, epsilon, 1.0 - epsilon)
    return np.log(values / (1.0 - values)).reshape(-1, 1)


class PlattCalibratedChurnClassifier:
    """
    Preserve the training estimator's churn-positive API while calibrating
    P(repeat purchase) with a Platt sigmoid learned from training OOF scores.

    predict_proba returns columns [P(repeat purchase), P(churn)] so column 1
    remains the SQL target class, churned=1.
    """

    def __init__(self, estimator, calibrator, epsilon: float = 1e-6):
        self.estimator = estimator
        self.calibrator = calibrator
        self.epsilon = epsilon
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X) -> np.ndarray:
        raw_churn_probability = np.asarray(
            self.estimator.predict_proba(X)[:, 1], dtype=float
        )
        raw_repeat_probability = 1.0 - raw_churn_probability
        calibrated_repeat_probability = self.calibrator.predict_proba(
            probability_logit(raw_repeat_probability, self.epsilon)
        )[:, 1]
        calibrated_repeat_probability = np.clip(
            calibrated_repeat_probability, 0.0, 1.0
        )
        calibrated_churn_probability = 1.0 - calibrated_repeat_probability
        return np.column_stack([
            calibrated_repeat_probability,
            calibrated_churn_probability,
        ])

    def predict(self, X) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def fit_platt_calibrator(
    repeat_probabilities: np.ndarray,
    repeat_targets: np.ndarray,
) -> LogisticRegression:
    """Fit a sigmoid calibrator on out-of-fold training predictions only."""
    scores = probability_logit(repeat_probabilities)
    targets = np.asarray(repeat_targets, dtype=int)
    if len(scores) != len(targets):
        raise ValueError("probabilities and targets must have the same length.")
    if set(np.unique(targets)) != {0, 1}:
        raise ValueError("calibration targets must contain both classes.")

    calibrator = LogisticRegression(
        C=1.0,
        solver="lbfgs",
        max_iter=2000,
        class_weight=None,
        random_state=42,
    )
    calibrator.fit(scores, targets)
    return calibrator
