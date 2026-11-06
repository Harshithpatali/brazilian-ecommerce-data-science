from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin


class UnivariateRepeatPurchaseBaseline(ClassifierMixin, BaseEstimator):
    """
    Point-in-time single-feature ranking baseline.

    The raw feature is monotonically transformed into a bounded score so it
    can be evaluated by the same repeat-purchase average-precision scorer and
    passed through the same training-only Platt calibration stage.
    """

    def __init__(
        self,
        feature_name: str,
        direction: str = "higher",
        transform: str = "log1p",
    ):
        self.feature_name = feature_name
        self.direction = direction
        self.transform = transform

    def _raw_score(self, X: pd.DataFrame) -> np.ndarray:
        if self.feature_name not in X.columns:
            raise ValueError(f"Missing baseline feature: {self.feature_name}")
        values = pd.to_numeric(
            X[self.feature_name], errors="coerce"
        ).fillna(0.0).to_numpy(dtype=float)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        if self.transform == "log1p":
            values = np.log1p(np.maximum(values, 0.0))
        elif self.transform != "identity":
            raise ValueError("transform must be 'log1p' or 'identity'.")
        if self.direction == "lower":
            values = -values
        elif self.direction != "higher":
            raise ValueError("direction must be 'higher' or 'lower'.")
        return values

    def fit(self, X: pd.DataFrame, y: np.ndarray):
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        if self.transform not in {"log1p", "identity"}:
            raise ValueError("transform must be 'log1p' or 'identity'.")
        values = self._raw_score(X)
        self.minimum_ = float(np.min(values))
        self.maximum_ = float(np.max(values))
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not hasattr(self, "minimum_"):
            raise RuntimeError("The baseline estimator has not been fitted.")
        values = self._raw_score(X)
        span = self.maximum_ - self.minimum_
        if span <= 0:
            repeat_probability = np.full(len(values), 0.5, dtype=float)
        else:
            repeat_probability = (values - self.minimum_) / span
        repeat_probability = np.clip(repeat_probability, 1e-6, 1.0 - 1e-6)
        return np.column_stack([repeat_probability, 1.0 - repeat_probability])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
