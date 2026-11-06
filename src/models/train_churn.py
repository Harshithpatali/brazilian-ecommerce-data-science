from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.churn_features import (
    CATEGORICAL_FEATURE_COLS,
    FEATURE_COLS,
    NUMERIC_FEATURE_COLS,
    add_churn_features,
)
from src.models.model_contract import MODEL_VERSION
from .mlflow_utils import configure_mlflow, get_tracking_uri

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ml" / "customer_churn_dataset.parquet"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

CAPACITY_FRACTION = 0.10
BOOTSTRAP_REPLICATES = 100


def _dates(values: pd.Series | list[Any]) -> pd.Series:
    result = pd.to_datetime(pd.Series(values), errors="coerce").dt.normalize()
    if result.isna().any():
        raise ValueError("snapshot_date contains invalid or missing values.")
    return result.reset_index(drop=True)


def chronological_partition_indices(
    snapshot_dates: pd.Series | list[Any],
    test_fraction: float = 0.15,
    validation_start_fraction: float = 0.65,
    horizon_days: int = 90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Make chronological train/validation/test partitions with label-window embargoes."""
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5.")
    if not 0 < validation_start_fraction < 0.7:
        raise ValueError("validation_start_fraction must be between 0 and 0.7.")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")

    row_dates = _dates(snapshot_dates)
    unique = pd.DatetimeIndex(sorted(row_dates.unique())).to_numpy(dtype="datetime64[ns]")
    if len(unique) < 12:
        raise ValueError("At least 12 unique dates are required for temporal evaluation.")

    test_start_idx = int(np.floor(len(unique) * (1 - test_fraction)))
    validation_start_idx = int(np.floor(len(unique) * validation_start_fraction))
    if validation_start_idx < 4 or test_start_idx - validation_start_idx < 4:
        raise ValueError("Not enough unique dates for partitions and embargo gaps.")

    validation_start_date = pd.Timestamp(unique[validation_start_idx])
    test_start_date = pd.Timestamp(unique[test_start_idx])
    horizon = pd.Timedelta(days=horizon_days)

    train_dates = unique[pd.to_datetime(unique) + horizon <= validation_start_date]
    validation_dates = unique[
        (pd.to_datetime(unique) >= validation_start_date)
        & (pd.to_datetime(unique) + horizon <= test_start_date)
    ]
    test_dates = unique[test_start_idx:]

    if len(train_dates) < 7 or len(validation_dates) < 2 or len(test_dates) < 2:
        raise ValueError(
            "Purging leaves too few dates for train, validation or final test. "
            "Use a longer observation history."
        )

    values = row_dates.to_numpy()
    indices = (
        np.flatnonzero(np.isin(values, train_dates)),
        np.flatnonzero(np.isin(values, validation_dates)),
        np.flatnonzero(np.isin(values, test_dates)),
    )
    periods = {
        "train_start": str(pd.Timestamp(train_dates[0]).date()),
        "train_end": str(pd.Timestamp(train_dates[-1]).date()),
        "validation_start": str(pd.Timestamp(validation_dates[0]).date()),
        "validation_end": str(pd.Timestamp(validation_dates[-1]).date()),
        "test_start": str(pd.Timestamp(test_dates[0]).date()),
        "test_end": str(pd.Timestamp(test_dates[-1]).date()),
        "label_horizon_embargo_days": str(horizon_days),
    }
    return *indices, periods


def make_date_grouped_splits(
    snapshot_dates: pd.Series | list[Any],
    y: pd.Series | np.ndarray | None = None,
    n_splits: int = 3,
    horizon_days: int = 90,
    min_positive_count: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Expanding-window CV grouped by snapshot date and purged by the label horizon.

    With sparse repeat events, candidate folds are rejected unless both fit and
    validation sides contain enough positive events. This avoids selecting
    models from folds with only one or two repeat-purchase examples.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    if min_positive_count < 1:
        raise ValueError("min_positive_count must be positive.")

    row_dates = _dates(snapshot_dates)
    unique = pd.DatetimeIndex(sorted(row_dates.unique())).to_numpy(dtype="datetime64[ns]")
    if n_splits < 2 or len(unique) < n_splits + 5:
        raise ValueError("Not enough unique dates for purged temporal CV.")

    values = row_dates.to_numpy()
    labels = None if y is None else np.asarray(pd.Series(y).reset_index(drop=True))
    if labels is not None and len(labels) != len(values):
        raise ValueError("y and snapshot_dates must have the same number of rows.")

    horizon = pd.Timedelta(days=horizon_days)
    minimum_warmup = max(5, int(np.ceil(len(unique) * 0.25)))

    for fold_count in range(min(n_splits, len(unique) - minimum_warmup), 1, -1):
        for warmup_end in range(minimum_warmup, len(unique) - fold_count + 1):
            remaining_dates = unique[warmup_end:]
            blocks = [
                block for block in np.array_split(remaining_dates, fold_count)
                if len(block)
            ]
            if len(blocks) != fold_count:
                continue

            candidate_folds: list[tuple[np.ndarray, np.ndarray]] = []
            valid_candidate = True
            for block in blocks:
                validation_start = pd.Timestamp(block[0])
                fit_dates = unique[
                    (pd.to_datetime(unique) + horizon <= validation_start)
                    & (unique < validation_start)
                ]
                fit_idx = np.flatnonzero(np.isin(values, fit_dates))
                validation_idx = np.flatnonzero(np.isin(values, block))

                if len(fit_idx) == 0 or len(validation_idx) == 0:
                    valid_candidate = False
                    break
                if labels is not None:
                    fit_positive = int(np.sum(labels[fit_idx] == 0))
                    val_positive = int(np.sum(labels[validation_idx] == 0))
                    fit_negative = int(np.sum(labels[fit_idx] == 1))
                    val_negative = int(np.sum(labels[validation_idx] == 1))
                    if min(fit_positive, val_positive) < min_positive_count:
                        valid_candidate = False
                        break
                    if min(fit_negative, val_negative) < min_positive_count:
                        valid_candidate = False
                        break
                candidate_folds.append((fit_idx, validation_idx))

            if valid_candidate:
                return candidate_folds

    raise ValueError(
        "Could not create purged temporal CV folds with enough repeat-purchase "
        "events in every training and validation block. Inspect the time range "
        "or reduce CV folds; do not replace temporal CV with random splitting."
    )


def repeat_purchase_average_precision_scorer(
    estimator: Any, X: pd.DataFrame, y_churned: pd.Series | np.ndarray
) -> float:
    """Average precision for the minority event: purchase within the next 90 days."""
    churn_probability = estimator.predict_proba(X)[:, 1]
    repeat_probability = 1.0 - churn_probability
    repeat_target = 1 - np.asarray(y_churned, dtype=int)
    return float(average_precision_score(repeat_target, repeat_probability))


def _logit(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


class CalibratedRepeatPurchaseModel:
    """
    Persistable wrapper that preserves the existing class order:
    class 0 = repeat purchase, class 1 = no purchase/churn.
    """

    def __init__(self, base_estimator: Any, calibrator: LogisticRegression):
        self.base_estimator = base_estimator
        self.calibrator = calibrator
        self.classes_ = np.array([0, 1])

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        raw_churn = self.base_estimator.predict_proba(X)[:, 1]
        raw_repeat = 1.0 - raw_churn
        calibrated_repeat = self.calibrator.predict_proba(_logit(raw_repeat))[:, 1]
        calibrated_repeat = np.clip(calibrated_repeat, 0.0, 1.0)
        return np.column_stack([calibrated_repeat, 1.0 - calibrated_repeat])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)


def assign_monthly_capacity(
    y_repeat: pd.Series | np.ndarray,
    repeat_scores: np.ndarray,
    snapshot_dates: pd.Series | np.ndarray,
    capacity_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    """Flag the highest-ranked customers within each monthly scoring snapshot."""
    if not 0 < capacity_fraction < 1:
        raise ValueError("capacity_fraction must be between 0 and 1.")

    y = np.asarray(y_repeat, dtype=int)
    scores = np.asarray(repeat_scores, dtype=float)
    dates = pd.to_datetime(pd.Series(snapshot_dates), errors="coerce").dt.normalize().to_numpy()
    if not (len(y) == len(scores) == len(dates)):
        raise ValueError("Labels, scores and snapshot dates must have equal lengths.")
    if np.isnan(scores).any() or pd.isna(dates).any():
        raise ValueError("Scores and snapshot dates must be non-missing.")

    selected = np.zeros(len(y), dtype=int)
    for date_value in np.unique(dates):
        indices = np.flatnonzero(dates == date_value)
        n_selected = max(1, int(np.ceil(len(indices) * capacity_fraction)))
        order = indices[np.argsort(-scores[indices], kind="mergesort")]
        selected[order[:n_selected]] = 1

    true_positive = int(np.sum((selected == 1) & (y == 1)))
    flagged = int(selected.sum())
    total_positive = int(y.sum())
    precision = true_positive / flagged if flagged else 0.0
    recall = true_positive / total_positive if total_positive else 0.0
    prevalence = float(y.mean()) if len(y) else 0.0
    return selected, {
        "precision": float(precision),
        "recall": float(recall),
        "lift": float(precision / prevalence) if prevalence > 0 else 0.0,
        "flagged": float(flagged),
        "flagged_fraction": float(flagged / len(y)) if len(y) else 0.0,
        "true_positives": float(true_positive),
        "total_positives": float(total_positive),
    }


def customer_bootstrap_pr_auc_ci(
    y_repeat: np.ndarray,
    scores: np.ndarray,
    customer_ids: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_REPLICATES,
    random_state: int = 42,
) -> dict[str, float]:
    """Customer-cluster bootstrap CI so repeated monthly rows are not treated as independent."""
    y = np.asarray(y_repeat, dtype=int)
    p = np.asarray(scores, dtype=float)
    ids = np.asarray(customer_ids).astype(str)
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    if len(unique_ids) < 2:
        raise ValueError("At least two unique customers are required for cluster bootstrap.")

    rng = np.random.default_rng(random_state)
    values: list[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.integers(0, len(unique_ids), size=len(unique_ids))
        multiplicity = np.bincount(sampled, minlength=len(unique_ids))
        weights = multiplicity[inverse]
        if np.sum(weights * y) == 0 or np.sum(weights * (1 - y)) == 0:
            continue
        values.append(float(average_precision_score(y, p, sample_weight=weights)))

    if len(values) < max(30, int(n_bootstrap * 0.5)):
        raise ValueError("Too few valid customer-bootstrap samples for a stable interval.")
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "lower_95": float(lower),
        "upper_95": float(upper),
        "replicates": int(len(values)),
    }


def _capacity_metrics_at_fractions(
    y_repeat: np.ndarray,
    scores: np.ndarray,
    snapshot_dates: pd.Series | np.ndarray,
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    metrics: dict[str, dict[str, float]] = {}
    selections: dict[str, np.ndarray] = {}
    for fraction in (0.05, 0.10, 0.20):
        selected, summary = assign_monthly_capacity(
            y_repeat, scores, snapshot_dates, capacity_fraction=fraction
        )
        key = f"{int(fraction * 100)}pct"
        metrics[key] = summary
        selections[key] = selected
    return metrics, selections


def _calibration_table(y_true: np.ndarray, probabilities: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame({
        "actual_repeat_purchase": np.asarray(y_true, dtype=int),
        "predicted_probability": np.asarray(probabilities, dtype=float),
    })
    frame["probability_decile"] = pd.qcut(
        frame["predicted_probability"], q=10, labels=False, duplicates="drop"
    )
    return (
        frame.groupby("probability_decile", dropna=True)
        .agg(
            customers=("actual_repeat_purchase", "size"),
            mean_predicted_probability=("predicted_probability", "mean"),
            observed_repeat_rate=("actual_repeat_purchase", "mean"),
        )
        .reset_index()
    )


def build_models() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)),
        ("scaler", StandardScaler()),
    ])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("one_hot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURE_COLS),
            ("categorical", categorical, CATEGORICAL_FEATURE_COLS),
        ],
        sparse_threshold=0.0,
    )
    return {
        "logistic_regression": (
            Pipeline([
                ("preprocess", preprocess),
                ("model", LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=42
                )),
            ]),
            {"model__C": [0.1, 1.0, 5.0], "model__solver": ["lbfgs"]},
        ),
        "hist_gradient_boosting": (
            Pipeline([
                ("preprocess", preprocess),
                ("model", HistGradientBoostingClassifier(
                    random_state=42, class_weight="balanced"
                )),
            ]),
            {
                "model__max_iter": [150, 300],
                "model__learning_rate": [0.05, 0.10],
                "model__max_leaf_nodes": [15, 31],
                "model__min_samples_leaf": [50],
                "model__l2_regularization": [1.0],
            },
        ),
        "random_forest": (
            Pipeline([
                ("preprocess", preprocess),
                ("model", RandomForestClassifier(
                    n_estimators=400,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                )),
            ]),
            {
                "model__max_depth": [4, 8],
                "model__min_samples_leaf": [5, 20],
                "model__max_features": ["sqrt", 0.8],
            },
        ),
        "catboost": (
            Pipeline([
                ("model", CatBoostClassifier(
                    iterations=350,
                    depth=6,
                    learning_rate=0.05,
                    loss_function="Logloss",
                    eval_metric="PRAUC",
                    auto_class_weights="Balanced",
                    cat_features=CATEGORICAL_FEATURE_COLS,
                    random_seed=42,
                    verbose=False,
                    allow_writing_files=False,
                    thread_count=1,
                )),
            ]),
            {
                "model__depth": [4, 6],
                "model__learning_rate": [0.05],
                "model__l2_leaf_reg": [3],
            },
        ),
    }


def _require_both_classes(y: pd.Series, name: str) -> None:
    if set(pd.Series(y).dropna().astype(int).unique()) != {0, 1}:
        raise ValueError(
            f"{name} must contain both classes; found "
            f"{pd.Series(y).value_counts(dropna=False).to_dict()}."
        )


def _distribution(y_churned: pd.Series) -> dict[str, int]:
    return {
        str(int(key)): int(value)
        for key, value in y_churned.value_counts().sort_index().items()
    }


def _top_capacity(y_repeat: np.ndarray, scores: np.ndarray, dates: pd.Series | np.ndarray):
    metrics, selections = _capacity_metrics_at_fractions(y_repeat, scores, dates)
    return metrics, selections


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Build the SQL feature table and run scripts.build_ml_data first.")

    df = pd.read_parquet(DATA)
    df = add_churn_features(df)
    required = {"snapshot_date", "customer_unique_id", "churned", *FEATURE_COLS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"Missing required repeat-purchase columns: {missing}")

    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df = df.dropna(subset=["snapshot_date", "customer_unique_id", "churned"])
    df["churned"] = df["churned"].astype(int)
    if not set(df["churned"].unique()).issubset({0, 1}):
        raise SystemExit("churned must contain only 0 and 1.")
    duplicates = int(df.duplicated(["customer_unique_id", "snapshot_date"]).sum())
    if duplicates:
        raise SystemExit(f"Found {duplicates} duplicate customer/snapshot rows.")

    df = df.sort_values(["snapshot_date", "customer_unique_id"]).reset_index(drop=True)
    y = df["churned"]
    X = df[FEATURE_COLS].copy()
    _require_both_classes(y, "Full dataset")

    train_idx, validation_idx, test_idx, periods = chronological_partition_indices(
        df["snapshot_date"]
    )
    X_train = X.iloc[train_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    X_validation = X.iloc[validation_idx].reset_index(drop=True)
    y_validation = y.iloc[validation_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)

    _require_both_classes(y_train, "Training partition")
    _require_both_classes(y_validation, "Validation partition")
    _require_both_classes(y_test, "Final test partition")

    train_dates = df["snapshot_date"].iloc[train_idx].reset_index(drop=True)
    n_dates = int(train_dates.dt.normalize().nunique())
    folds = make_date_grouped_splits(
        train_dates,
        y=y_train,
        n_splits=min(3, n_dates - 5),
        min_positive_count=15,
    )

    experiment = configure_mlflow()
    searches: dict[str, GridSearchCV] = {}
    rows: list[dict[str, Any]] = []

    for name, (pipeline, grid) in build_models().items():
        with mlflow.start_run(run_name=f"cv_tuned_{name}"):
            search = GridSearchCV(
                pipeline,
                grid,
                scoring=repeat_purchase_average_precision_scorer,
                cv=folds,
                n_jobs=2,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            searches[name] = search
            mlflow.log_params({
                "model_family": name,
                "cv": "expanding_window_grouped_by_snapshot_date_with_label_purge",
                "cv_folds": len(folds),
                "selection_metric": "cv_average_precision_for_repeat_purchase_90d",
                "model_version": MODEL_VERSION,
            })
            mlflow.log_params({
                key: str(value) for key, value in search.best_params_.items()
            })
            mlflow.log_metric("cv_repeat_purchase_pr_auc", float(search.best_score_))
            rows.append({
                "model": name,
                "cv_repeat_purchase_pr_auc": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            })

    comparison = (
        pd.DataFrame(rows)
        .sort_values(
            ["cv_repeat_purchase_pr_auc", "model"],
            ascending=[False, True],
        )
        .reset_index(drop=True)
    )
    selected_name = str(comparison.iloc[0]["model"])
    selected_search = searches[selected_name]
    selected_estimator = selected_search.best_estimator_

    # Fit Platt scaling on the strictly later validation period, not on training
    # or the final holdout. The base estimator is frozen after CV selection.
    validation_raw_repeat = 1.0 - selected_estimator.predict_proba(X_validation)[:, 1]
    y_validation_repeat = 1 - y_validation.to_numpy(dtype=int)
    calibrator = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    calibrator.fit(_logit(validation_raw_repeat), y_validation_repeat)
    calibrated_validation = calibrator.predict_proba(_logit(validation_raw_repeat))[:, 1]

    repeat_threshold = float(np.quantile(calibrated_validation, 1.0 - CAPACITY_FRACTION))
    churn_threshold = float(np.nextafter(1.0 - repeat_threshold, np.inf))
    validation_capacity, _ = assign_monthly_capacity(
        y_validation_repeat,
        calibrated_validation,
        df.iloc[validation_idx]["snapshot_date"].reset_index(drop=True),
        CAPACITY_FRACTION,
    )

    model = CalibratedRepeatPurchaseModel(selected_estimator, calibrator)
    test_prob = model.predict_proba(X_test)[:, 1]
    test_repeat_prob = 1.0 - test_prob
    y_test_repeat = 1 - y_test.to_numpy(dtype=int)
    test_dates = df.iloc[test_idx]["snapshot_date"].reset_index(drop=True)

    capacity_metrics, capacity_predictions = _top_capacity(
        y_test_repeat, test_repeat_prob, test_dates
    )
    predicted_repeat = capacity_predictions["10pct"]
    test_pred = 1 - predicted_repeat

    test_ap = float(average_precision_score(y_test_repeat, test_repeat_prob))
    test_roc_auc = float(roc_auc_score(y_test_repeat, test_repeat_prob))
    test_prevalence = float(y_test_repeat.mean())
    train_prevalence = float((1 - y_train.to_numpy(dtype=int)).mean())
    dummy_prob = np.full(len(y_test_repeat), train_prevalence, dtype=float)
    dummy_ap = float(average_precision_score(y_test_repeat, dummy_prob))
    dummy_brier = float(brier_score_loss(y_test_repeat, dummy_prob))
    model_brier = float(brier_score_loss(y_test_repeat, test_repeat_prob))
    recency_score = -X_test["recency_days"].to_numpy(dtype=float)
    recency_ap = float(average_precision_score(y_test_repeat, recency_score))

    test_metrics: dict[str, float] = {
        "roc_auc": test_roc_auc,
        "pr_auc": test_ap,
        "precision": float(precision_score(y_test_repeat, predicted_repeat, zero_division=0)),
        "recall": float(recall_score(y_test_repeat, predicted_repeat, zero_division=0)),
        "f1": float(f1_score(y_test_repeat, predicted_repeat, zero_division=0)),
        "f2": float(fbeta_score(y_test_repeat, predicted_repeat, beta=2.0, zero_division=0)),
        "class_0_precision": float(precision_score(y_test, test_pred, pos_label=0, zero_division=0)),
        "class_0_recall": float(recall_score(y_test, test_pred, pos_label=0, zero_division=0)),
        "class_0_f1": float(f1_score(y_test, test_pred, pos_label=0, zero_division=0)),
        "class_1_precision": float(precision_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "class_1_recall": float(recall_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "class_1_f1": float(f1_score(y_test, test_pred, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred)),
        "matthews_corrcoef": float(matthews_corrcoef(y_test_repeat, predicted_repeat)),
        "positive_prevalence": test_prevalence,
        "threshold": churn_threshold,
        "dummy_baseline_pr_auc": dummy_ap,
        "pr_auc_lift_over_dummy": float(test_ap - dummy_ap),
        "recency_only_pr_auc": recency_ap,
        "pr_auc_lift_over_recency": float(test_ap / recency_ap) if recency_ap > 0 else 0.0,
        "brier_score": model_brier,
        "log_loss": float(log_loss(y_test_repeat, test_repeat_prob, labels=[0, 1])),
        "brier_skill_score": float(1.0 - model_brier / dummy_brier) if dummy_brier > 0 else 0.0,
    }
    for fraction_key, summary in capacity_metrics.items():
        test_metrics[f"precision_at_{fraction_key}"] = summary["precision"]
        test_metrics[f"recall_at_{fraction_key}"] = summary["recall"]
        test_metrics[f"lift_at_{fraction_key}"] = summary["lift"]
        test_metrics[f"flagged_fraction_at_{fraction_key}"] = summary["flagged_fraction"]

    bootstrap_ci = customer_bootstrap_pr_auc_ci(
        y_test_repeat,
        test_repeat_prob,
        df.iloc[test_idx]["customer_unique_id"].to_numpy(),
    )

    predictions = pd.DataFrame({
        "customer_unique_id": df.iloc[test_idx]["customer_unique_id"].to_numpy(),
        "snapshot_date": test_dates.to_numpy(),
        "actual_churned": y_test.to_numpy(),
        "actual_repeat_purchase": y_test_repeat,
        "churn_probability": test_prob,
        "repeat_purchase_probability": test_repeat_prob,
        "predicted_repeat_top_5pct": capacity_predictions["5pct"],
        "predicted_repeat_top_10pct": capacity_predictions["10pct"],
        "predicted_repeat_top_20pct": capacity_predictions["20pct"],
    })
    predictions.to_parquet(MODEL_DIR / "test_predictions.parquet", index=False)

    precision, recall, pr_thresholds = precision_recall_curve(y_test_repeat, test_repeat_prob)
    pd.DataFrame({
        "precision": precision[:-1],
        "recall": recall[:-1],
        "repeat_purchase_probability_threshold": pr_thresholds,
    }).to_csv(MODEL_DIR / "test_precision_recall_curve.csv", index=False)
    _calibration_table(y_test_repeat, test_repeat_prob).to_csv(
        MODEL_DIR / "test_calibration.csv", index=False
    )

    report_path = MODEL_DIR / "test_classification_report.json"
    report_path.write_text(json.dumps(classification_report(
        y_test,
        test_pred,
        labels=[0, 1],
        target_names=["purchase_within_90_days", "no_purchase_within_90_days"],
        output_dict=True,
        zero_division=0,
    ), indent=2), encoding="utf-8")

    matrix_path = MODEL_DIR / "test_confusion_matrix.json"
    matrix_path.write_text(json.dumps({
        "labels": ["purchase_within_90_days", "no_purchase_within_90_days"],
        "matrix": confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist(),
        "policy": "top_10_percent_per_snapshot_date",
        "capacity_fraction": CAPACITY_FRACTION,
    }, indent=2), encoding="utf-8")

    comparison["selected"] = comparison["model"].eq(selected_name)
    comparison_path = MODEL_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    model_path = MODEL_DIR / "best_churn_model.joblib"
    joblib.dump(model, model_path)

    metadata = {
        "pipeline_status": "ready",
        "model_version": MODEL_VERSION,
        "experiment_name": experiment,
        "tracking_uri": get_tracking_uri(),
        "selected_model": selected_name,
        "selection_metric": "purged_temporal_cv_average_precision_for_repeat_purchase_90d",
        "best_params": selected_search.best_params_,
        "threshold": churn_threshold,
        "repeat_purchase_probability_threshold": repeat_threshold,
        "threshold_selection": (
            "validation 90th percentile calibrated repeat-purchase probability; "
            "operational ranking selects the top 10 percent per monthly snapshot"
        ),
        "threshold_policy": "top_10_percent_per_snapshot_date",
        "capacity_fraction": CAPACITY_FRACTION,
        "calibration_method": "Platt sigmoid fitted on separate chronological validation period",
        "calibration_intercept": float(calibrator.intercept_[0]),
        "calibration_slope": float(calibrator.coef_[0][0]),
        "validation_threshold_metrics": {
            "capacity_fraction": CAPACITY_FRACTION,
            "precision": float(precision_score(y_validation_repeat, validation_capacity, zero_division=0)),
            "recall": float(recall_score(y_validation_repeat, validation_capacity, zero_division=0)),
            "flagged_fraction": float(validation_capacity.mean()),
        },
        "evaluation_positive_class": "purchase_within_90_days (SQL churned=0)",
        "target_definition": (
            "churned=1 when no eventually-delivered order has a purchase timestamp in "
            "[snapshot_date, snapshot_date + 90 days); repeat purchase is churned=0."
        ),
        "population_definition": (
            "Monthly customer snapshots after each customer's first delivered order, "
            "when at least one delivered order was purchased and delivered in the prior 180 days."
        ),
        "feature_columns": FEATURE_COLS,
        "feature_availability": {
            "historical_orders": (
                "order_purchase_timestamp < snapshot_date and "
                "order_delivered_customer_date < snapshot_date"
            ),
            "review_score": "review_answer_timestamp < snapshot_date",
            "customer_state": "state from earliest delivered order observable before snapshot_date",
            "snapshot_month": "calendar month of snapshot_date",
            "lifetime_history": (
                "purchase_timestamp < snapshot_date and delivered_timestamp < snapshot_date"
            ),
            "prior_order_window": "[snapshot_date - 360 days, snapshot_date - 180 days)",
            "future_orders": "used only to construct the target, never as predictors",
        },
        "feature_window_days": 180,
        "max_historical_lookback_days": 360,
        "prediction_window_days": 90,
        "label_horizon_embargo_days": 90,
        "evaluation_design": (
            "Customer-month decisions are split chronologically by unique snapshot date. "
            "Each partition boundary has a 90-day purge. CV uses expanding windows grouped "
            "by snapshot date, with minimum positive-event counts in each fold. Customers "
            "can appear in multiple months, matching known-customer monthly CRM scoring. "
            "Uncertainty intervals use customer-cluster bootstrap."
        ),
        "periods": periods,
        "class_distribution": {
            "all": _distribution(y),
            "train": _distribution(y_train),
            "validation": _distribution(y_validation),
            "test": _distribution(y_test),
        },
        "partition_rows": {
            "all": int(len(df)),
            "train": int(len(train_idx)),
            "validation": int(len(validation_idx)),
            "test": int(len(test_idx)),
        },
        "unique_customers": {
            "all": int(df["customer_unique_id"].nunique()),
            "train": int(df.iloc[train_idx]["customer_unique_id"].nunique()),
            "validation": int(df.iloc[validation_idx]["customer_unique_id"].nunique()),
            "test": int(df.iloc[test_idx]["customer_unique_id"].nunique()),
        },
        "test_metrics": test_metrics,
        "customer_bootstrap_pr_auc_95_ci": bootstrap_ci,
        "dummy_baseline": {
            "strategy": "training_prevalence",
            "train_positive_prevalence": train_prevalence,
            "test_pr_auc": dummy_ap,
            "test_positive_prevalence": test_prevalence,
            "test_brier_score": dummy_brier,
            "test_log_loss": float(log_loss(y_test_repeat, dummy_prob, labels=[0, 1])),
            "recency_only_pr_auc": recency_ap,
        },
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "artifacts": {
            "model": "models/best_churn_model.joblib",
            "comparison": "models/model_comparison.csv",
            "test_predictions": "models/test_predictions.parquet",
            "classification_report": "models/test_classification_report.json",
            "confusion_matrix": "models/test_confusion_matrix.json",
            "precision_recall_curve": "models/test_precision_recall_curve.csv",
            "calibration": "models/test_calibration.csv",
        },
    }
    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    with mlflow.start_run(run_name="final_selected_model_holdout"):
        mlflow.log_params({
            "selected_model": selected_name,
            "selection_metric": "purged_temporal_cv_average_precision_for_repeat_purchase_90d",
            "model_version": MODEL_VERSION,
            "threshold_policy": "top_10_percent_per_snapshot_date",
            "test_start": periods["test_start"],
            "test_end": periods["test_end"],
        })
        mlflow.log_metrics({
            f"test_{key}": float(value)
            for key, value in test_metrics.items()
            if isinstance(value, (float, int)) and np.isfinite(value)
        })
        for artifact, folder in [
            (comparison_path, "model_selection"),
            (metadata_path, "model_selection"),
            (report_path, "evaluation"),
            (matrix_path, "evaluation"),
            (MODEL_DIR / "test_precision_recall_curve.csv", "evaluation"),
            (MODEL_DIR / "test_calibration.csv", "evaluation"),
            (MODEL_DIR / "test_predictions.parquet", "evaluation"),
        ]:
            mlflow.log_artifact(str(artifact), artifact_path=folder)
        mlflow.sklearn.log_model(
            selected_estimator, artifact_path="base_estimator", input_example=X_train.head(3)
        )

    print("Target distribution (1 = no delivered purchase; 0 = purchase within 90 days):")
    print(y.value_counts(normalize=True).sort_index().to_string())
    print("Periods:", json.dumps(periods, indent=2))
    print("Temporal CV model comparison (repeat-purchase PR-AUC):")
    print(comparison.to_string(index=False))
    print("Selected model:", selected_name)
    print("Calibration method: validation-fitted Platt sigmoid")
    print("Operational policy: top 10% by repeat-purchase score in each monthly snapshot")
    print("Final untouched test metrics:", json.dumps(test_metrics, indent=2))
    print("Customer-bootstrap PR-AUC 95% CI:", json.dumps(bootstrap_ci))


if __name__ == "__main__":
    main()
