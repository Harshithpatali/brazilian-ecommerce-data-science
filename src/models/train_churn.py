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
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, brier_score_loss, classification_report, confusion_matrix,
    balanced_accuracy_score, f1_score, fbeta_score, log_loss, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features.churn_features import FEATURE_COLS
from src.models.model_contract import MODEL_VERSION
from src.models.baselines import UnivariateRepeatPurchaseBaseline
from src.models.calibration import (
    PlattCalibratedChurnClassifier,
    fit_platt_calibrator,
)
from .mlflow_utils import configure_mlflow, get_tracking_uri

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ml" / "customer_churn_dataset.parquet"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)



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
    """
    Split unique dates chronologically and purge target-window overlap.

    A training snapshot is retained only if its full forward label window
    ends on or before the validation period begins. Validation snapshots are
    retained only if their full label window ends on or before final test
    begins. Rows in the embargo gaps are deliberately excluded.
    """
    if not 0 < test_fraction < 0.5:
        raise ValueError("test_fraction must be between 0 and 0.5.")
    if not 0 < validation_start_fraction < 0.7:
        raise ValueError("validation_start_fraction must be between 0 and 0.7.")
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")

    row_dates = _dates(snapshot_dates)
    unique = pd.DatetimeIndex(sorted(row_dates.unique())).to_numpy(dtype="datetime64[ns]")
    if len(unique) < 12:
        raise ValueError(
            "At least 12 unique snapshot dates are required for purged "
            "train/validation/test periods and temporal CV."
        )

    test_start_idx = int(np.floor(len(unique) * (1 - test_fraction)))
    validation_start_idx = int(np.floor(len(unique) * validation_start_fraction))
    if validation_start_idx < 4 or test_start_idx - validation_start_idx < 4:
        raise ValueError(
            "Not enough unique dates for chronological partitions and 90-day embargo gaps."
        )

    validation_start_date = pd.Timestamp(unique[validation_start_idx])
    test_start_date = pd.Timestamp(unique[test_start_idx])
    horizon = pd.Timedelta(days=horizon_days)

    train_dates = unique[
        pd.to_datetime(unique) + horizon <= validation_start_date
    ]
    validation_dates = unique[
        (pd.to_datetime(unique) >= validation_start_date)
        & (pd.to_datetime(unique) + horizon <= test_start_date)
    ]
    test_dates = unique[test_start_idx:]

    if len(train_dates) < 7 or len(validation_dates) < 2 or len(test_dates) < 2:
        raise ValueError(
            "90-day purging leaves too few dates for temporal CV, validation, "
            "or final testing. At least 7 training dates, 2 validation dates, "
            "and 2 test dates are required. Use a longer observation history."
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
    n_splits: int = 4,
    horizon_days: int = 90,
    min_train_repeat_events: int = 1,
    min_validation_repeat_events: int = 1,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build expanding-window folds over unique dates with a purged label gap.

    The latest training snapshot in a fold must have its full forward target
    window observed by the first validation snapshot. If labels are supplied,
    reduce fold count/adjust the warm-up until every side contains both classes.
    """
    if horizon_days <= 0:
        raise ValueError("horizon_days must be positive.")
    row_dates = _dates(snapshot_dates)
    unique = pd.DatetimeIndex(sorted(row_dates.unique())).to_numpy(dtype="datetime64[ns]")
    if n_splits < 2 or len(unique) < n_splits + 5:
        raise ValueError("Not enough unique dates for purged temporal CV.")
    if min_train_repeat_events < 1 or min_validation_repeat_events < 1:
        raise ValueError("Minimum repeat-purchase event counts must be positive.")

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

            candidate_folds = []
            valid_candidate = True
            for block in blocks:
                validation_start = pd.Timestamp(block[0])
                fit_dates = unique[
                    pd.to_datetime(unique) + horizon <= validation_start
                ]
                validation_dates = block

                # Only use fitting dates that precede this block, not any later
                # dates that happen to be in the complete training partition.
                fit_dates = fit_dates[fit_dates < validation_start]
                fit_idx = np.flatnonzero(np.isin(values, fit_dates))
                validation_idx = np.flatnonzero(
                    np.isin(values, validation_dates)
                )

                if len(fit_idx) == 0 or len(validation_idx) == 0:
                    valid_candidate = False
                    break
                if labels is not None:
                    if len(np.unique(labels[fit_idx])) != 2:
                        valid_candidate = False
                        break
                    if len(np.unique(labels[validation_idx])) != 2:
                        valid_candidate = False
                        break
                    # SQL churned=0 is the second-purchase event. Reject folds
                    # whose minority-event support is too small for meaningful AP.
                    if int((labels[fit_idx] == 0).sum()) < min_train_repeat_events:
                        valid_candidate = False
                        break
                    if int((labels[validation_idx] == 0).sum()) < min_validation_repeat_events:
                        valid_candidate = False
                        break
                candidate_folds.append((fit_idx, validation_idx))

            if valid_candidate:
                return candidate_folds

    raise ValueError(
        "Could not create at least two purged chronological CV folds containing "
        "both classes in training and validation. Inspect monthly class prevalence "
        "or use a longer observation history."
    )

def repeat_purchase_average_precision_scorer(
    estimator: Any, X: pd.DataFrame, y_churned: pd.Series | np.ndarray
) -> float:
    """
    Score the minority event: a delivered purchase within 90 days.

    The estimator is fitted to churned=1, so its class-1 probability is
    converted to P(repeat purchase) = 1 - P(churn).
    """
    churn_probability = estimator.predict_proba(X)[:, 1]
    repeat_probability = 1.0 - churn_probability
    repeat_target = 1 - np.asarray(y_churned, dtype=int)
    return float(average_precision_score(repeat_target, repeat_probability))


def choose_fbeta_threshold(
    y_churned: pd.Series | np.ndarray,
    churn_probabilities: np.ndarray,
    beta: float = 2.0,
) -> tuple[float, dict[str, float]]:
    """
    Select a recall-oriented repeat-purchase threshold using validation F-beta.

    F2 weights recall more heavily than precision. The returned threshold is
    the complementary churn-probability cutoff used by the Streamlit risk UI.
    """
    if not np.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a finite positive number.")

    repeat_target = 1 - np.asarray(y_churned, dtype=int)
    repeat_probabilities = 1.0 - np.asarray(churn_probabilities, dtype=float)
    precision, recall, thresholds = precision_recall_curve(
        repeat_target, repeat_probabilities
    )
    if not len(thresholds):
        repeat_threshold = 0.5
    else:
        beta_squared = beta ** 2
        denominator = beta_squared * precision[:-1] + recall[:-1]
        fbeta_values = np.divide(
            (1 + beta_squared) * precision[:-1] * recall[:-1],
            denominator,
            out=np.zeros_like(denominator),
            where=denominator > 0,
        )
        repeat_threshold = float(thresholds[int(np.nanargmax(fbeta_values))])

    predicted_repeat = (repeat_probabilities >= repeat_threshold).astype(int)
    churn_threshold = min(1.0, float(np.nextafter(1.0 - repeat_threshold, np.inf)))
    return churn_threshold, {
        "precision": float(precision_score(repeat_target, predicted_repeat, zero_division=0)),
        "recall": float(recall_score(repeat_target, predicted_repeat, zero_division=0)),
        "f1": float(f1_score(repeat_target, predicted_repeat, zero_division=0)),
        "f2": float(fbeta_score(repeat_target, predicted_repeat, beta=2.0, zero_division=0)),
        "beta": float(beta),
        "repeat_purchase_probability_threshold": repeat_threshold,
    }


def build_models() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
        ("scaler", StandardScaler()),
    ])
    preprocess = ColumnTransformer([("numeric", numeric, FEATURE_COLS)])
    return {
        "logistic_regression": (
            Pipeline([
                ("preprocess", preprocess),
                ("model", LogisticRegression(
                    max_iter=3000, class_weight="balanced", random_state=42
                )),
            ]),
            {"model__C": [0.05, 0.2, 1.0, 5.0], "model__solver": ["liblinear"]},
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
                "model__learning_rate": [0.03, 0.07],
                "model__max_leaf_nodes": [15, 31],
                "model__min_samples_leaf": [100],
                "model__l2_regularization": [1.0],
            },
        ),
        "random_forest": (
            Pipeline([
                ("preprocess", preprocess),
                ("model", RandomForestClassifier(
                    n_estimators=120,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=1,
                )),
            ]),
            {
                "model__max_depth": [6, None],
                "model__min_samples_leaf": [20, 50],
                "model__max_features": ["sqrt"],
            },
        ),
        "baseline_recency_only": (
            UnivariateRepeatPurchaseBaseline(
                "recency_days", direction="lower", transform="log1p"
            ),
            {},
        ),
        "baseline_recent_frequency_only": (
            UnivariateRepeatPurchaseBaseline(
                "orders_180d", direction="higher", transform="log1p"
            ),
            {},
        ),
        "baseline_lifetime_frequency_only": (
            UnivariateRepeatPurchaseBaseline(
                "lifetime_delivered_orders",
                direction="higher",
                transform="log1p",
            ),
            {},
        ),
    }


def _require_both_classes(y: pd.Series, name: str) -> None:
    if set(pd.Series(y).dropna().astype(int).unique()) != {0, 1}:
        raise ValueError(
            f"{name} must contain both classes; found "
            f"{pd.Series(y).value_counts(dropna=False).to_dict()}. "
            "Inspect the target and date coverage before training."
        )


def _distribution(y: pd.Series) -> dict[str, int]:
    return {str(int(k)): int(v) for k, v in y.value_counts().sort_index().items()}




def _fit_oof_platt_calibrator(
    estimator: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[Any, int]:
    """Fit calibration using purged out-of-fold training predictions only."""
    oof_repeat_probability = np.full(len(y_train), np.nan, dtype=float)
    for fit_idx, calibration_idx in folds:
        fold_estimator = clone(estimator)
        fold_estimator.fit(X_train.iloc[fit_idx], y_train.iloc[fit_idx])
        fold_churn_probability = fold_estimator.predict_proba(
            X_train.iloc[calibration_idx]
        )[:, 1]
        oof_repeat_probability[calibration_idx] = 1.0 - fold_churn_probability

    valid = np.isfinite(oof_repeat_probability)
    oof_repeat_target = 1 - y_train.to_numpy(dtype=int)
    calibrator = fit_platt_calibrator(
        oof_repeat_probability[valid],
        oof_repeat_target[valid],
    )
    return calibrator, int(valid.sum())


def _expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> float:
    """Expected calibration error using equal-width probability bins."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = max(len(y), 1)
    error = 0.0
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        if index == n_bins - 1:
            mask = (p >= lower) & (p <= upper)
        else:
            mask = (p >= lower) & (p < upper)
        if mask.any():
            error += float(mask.sum()) / total * abs(
                float(p[mask].mean()) - float(y[mask].mean())
            )
    return float(error)


def _calibration_bins(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Return reliability-diagram values for the final test set."""
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for index in range(n_bins):
        lower, upper = edges[index], edges[index + 1]
        mask = ((p >= lower) & (p <= upper)) if index == n_bins - 1 else (
            (p >= lower) & (p < upper)
        )
        rows.append({
            "bin": index + 1,
            "lower_probability": lower,
            "upper_probability": upper,
            "count": int(mask.sum()),
            "mean_predicted_probability": float(p[mask].mean()) if mask.any() else np.nan,
            "observed_purchase_rate": float(y[mask].mean()) if mask.any() else np.nan,
        })
    return pd.DataFrame(rows)


def _monthly_capacity_metrics(
    y_true: np.ndarray,
    repeat_probabilities: np.ndarray,
    snapshot_dates: pd.Series | np.ndarray,
    capacity_fraction: float,
) -> dict[str, float | int]:
    """
    Rank customers inside each monthly campaign cohort, then aggregate results.

    This is more operationally meaningful than applying one F2 threshold that
    can send a very large and unstable share of customers to a campaign.
    """
    if not 0 < capacity_fraction < 1:
        raise ValueError("capacity_fraction must be between 0 and 1.")
    frame = pd.DataFrame({
        "snapshot_date": pd.to_datetime(pd.Series(snapshot_dates)).dt.normalize(),
        "actual_repeat": np.asarray(y_true, dtype=int),
        "repeat_probability": np.asarray(repeat_probabilities, dtype=float),
    })
    selected_count = 0
    selected_positive = 0
    total_positive = int(frame["actual_repeat"].sum())

    for _, cohort in frame.groupby("snapshot_date", sort=True):
        n_select = max(1, int(np.ceil(len(cohort) * capacity_fraction)))
        selected = cohort.nlargest(n_select, "repeat_probability")
        selected_count += len(selected)
        selected_positive += int(selected["actual_repeat"].sum())

    precision_at_capacity = (
        selected_positive / selected_count if selected_count else 0.0
    )
    prevalence = total_positive / len(frame) if len(frame) else 0.0
    return {
        "capacity_fraction": float(capacity_fraction),
        "monthly_cohorts": int(frame["snapshot_date"].nunique()),
        "snapshots_scored": int(len(frame)),
        "customers_contacted": int(selected_count),
        "true_repeat_purchases_captured": int(selected_positive),
        "precision": float(precision_at_capacity),
        "recall": float(selected_positive / total_positive) if total_positive else 0.0,
        "lift_over_monthly_prevalence": (
            float(precision_at_capacity / prevalence) if prevalence > 0 else 0.0
        ),
        "overall_repeat_purchase_prevalence": float(prevalence),
    }


def _cluster_bootstrap_pr_auc_ci(
    y_true: np.ndarray,
    repeat_probabilities: np.ndarray,
    customer_ids: pd.Series | np.ndarray,
    n_bootstrap: int = 200,
    random_state: int = 42,
) -> dict[str, float | int]:
    """
    Estimate a 95% PR-AUC interval by resampling customers, not correlated rows.
    """
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(repeat_probabilities, dtype=float)
    group_codes, unique_groups = pd.factorize(
        pd.Series(customer_ids).astype(str), sort=False
    )
    if len(unique_groups) < 2:
        return {"lower_95": float("nan"), "upper_95": float("nan"), "valid_replicates": 0}

    order = np.argsort(group_codes)
    counts = np.bincount(group_codes, minlength=len(unique_groups))
    offsets = np.concatenate([[0], np.cumsum(counts)])
    group_indices = [
        order[offsets[i]:offsets[i + 1]] for i in range(len(unique_groups))
    ]
    rng = np.random.default_rng(random_state)
    values: list[float] = []

    for _ in range(n_bootstrap):
        chosen_groups = rng.integers(0, len(unique_groups), size=len(unique_groups))
        sampled_indices = np.concatenate([group_indices[i] for i in chosen_groups])
        sampled_y = y[sampled_indices]
        if len(np.unique(sampled_y)) < 2:
            continue
        values.append(float(average_precision_score(
            sampled_y, p[sampled_indices]
        )))

    if not values:
        return {"lower_95": float("nan"), "upper_95": float("nan"), "valid_replicates": 0}
    lower, upper = np.quantile(values, [0.025, 0.975])
    return {
        "lower_95": float(lower),
        "upper_95": float(upper),
        "valid_replicates": int(len(values)),
    }


def _ranking_summary(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int | None]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    result: dict[str, float | int | None] = {
        "rows": int(len(y)),
        "positive_count": int(y.sum()),
        "positive_prevalence": float(y.mean()) if len(y) else 0.0,
    }
    if len(np.unique(y)) == 2:
        result["pr_auc"] = float(average_precision_score(y, p))
        result["roc_auc"] = float(roc_auc_score(y, p))
    else:
        result["pr_auc"] = None
        result["roc_auc"] = None
    return result


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Build the SQL feature table and run scripts.build_ml_data first.")

    df = pd.read_parquet(DATA)
    required = {"snapshot_date", "customer_unique_id", "churned", *FEATURE_COLS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise SystemExit(f"Missing required churn columns: {missing}")

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
    if len(df) < 300_000 or df["customer_unique_id"].nunique() < 70_000:
        raise SystemExit(
            "The first-purchase cohort is unexpectedly small. Expected the full "
            "Olist extract (>=300,000 customer-month rows and >=70,000 customers); "
            f"received {len(df):,} rows and {df['customer_unique_id'].nunique():,} customers. "
            "Check the database connection and snapshot cohort rebuild."
        )

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
        n_splits=min(4, n_dates - 5),
        min_train_repeat_events=10,
        min_validation_repeat_events=5,
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
                pre_dispatch=2,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            searches[name] = search
            mlflow.log_params({
                "model_family": name,
                "cv": "expanding_window_grouped_by_snapshot_date",
                "cv_folds": len(folds),
                "selection_metric": "cv_average_precision_for_second_purchase_within_90_days_against_rfm_baselines_against_RFM_baselines",
                "model_version": MODEL_VERSION,
            })
            mlflow.log_params({
                key: str(value) for key, value in search.best_params_.items()
            })
            mlflow.log_metric(
                "cv_repeat_purchase_pr_auc", float(search.best_score_)
            )
            rows.append({
                "model": name,
                "cv_repeat_purchase_pr_auc": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            })

    comparison = pd.DataFrame(rows).sort_values(
        ["cv_repeat_purchase_pr_auc", "model"], ascending=[False, True]
    ).reset_index(drop=True)
    selected_name = str(comparison.iloc[0]["model"])
    selected_search = searches[selected_name]
    selected_model = selected_search.best_estimator_

    # The sigmoid calibrator is trained on purged OOF predictions from training
    # rows only. Validation selects the operating threshold; test stays untouched.
    calibrator, calibration_rows = _fit_oof_platt_calibrator(
        selected_model, X_train, y_train, folds
    )
    calibrated_model = PlattCalibratedChurnClassifier(
        selected_model, calibrator
    )

    validation_churn_probability = calibrated_model.predict_proba(
        X_validation
    )[:, 1]
    threshold, validation_threshold_metrics = choose_fbeta_threshold(
        y_validation, validation_churn_probability, beta=2.0
    )

    test_churn_probability = calibrated_model.predict_proba(X_test)[:, 1]
    test_repeat_probability = 1.0 - test_churn_probability
    y_test_repeat = 1 - y_test.to_numpy(dtype=int)
    repeat_threshold = float(
        validation_threshold_metrics["repeat_purchase_probability_threshold"]
    )
    test_pred_repeat = (test_repeat_probability >= repeat_threshold).astype(int)
    test_pred_churn = 1 - test_pred_repeat

    test_metrics: dict[str, float] = {
        "roc_auc": float(roc_auc_score(y_test_repeat, test_repeat_probability)),
        "pr_auc": float(average_precision_score(y_test_repeat, test_repeat_probability)),
        "precision": float(precision_score(y_test_repeat, test_pred_repeat, zero_division=0)),
        "recall": float(recall_score(y_test_repeat, test_pred_repeat, zero_division=0)),
        "f1": float(f1_score(y_test_repeat, test_pred_repeat, zero_division=0)),
        "f2": float(fbeta_score(y_test_repeat, test_pred_repeat, beta=2.0, zero_division=0)),
        "class_0_precision": float(precision_score(y_test, test_pred_churn, pos_label=0, zero_division=0)),
        "class_0_recall": float(recall_score(y_test, test_pred_churn, pos_label=0, zero_division=0)),
        "class_0_f1": float(f1_score(y_test, test_pred_churn, pos_label=0, zero_division=0)),
        "class_1_precision": float(precision_score(y_test, test_pred_churn, pos_label=1, zero_division=0)),
        "class_1_recall": float(recall_score(y_test, test_pred_churn, pos_label=1, zero_division=0)),
        "class_1_f1": float(f1_score(y_test, test_pred_churn, pos_label=1, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, test_pred_churn)),
        "matthews_corrcoef": float(matthews_corrcoef(y_test_repeat, test_pred_repeat)),
        "positive_prevalence": float(y_test_repeat.mean()),
        "threshold": float(threshold),
        "brier_score": float(brier_score_loss(y_test_repeat, test_repeat_probability)),
        "log_loss": float(log_loss(
            y_test_repeat,
            np.clip(test_repeat_probability, 1e-9, 1.0 - 1e-9),
            labels=[0, 1],
        )),
        "expected_calibration_error": _expected_calibration_error(
            y_test_repeat, test_repeat_probability, n_bins=10
        ),
    }

    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    dummy_churn_probability = dummy.predict_proba(X_test)[:, 1]
    dummy_repeat_probability = 1.0 - dummy_churn_probability
    dummy_pr_auc = float(average_precision_score(
        y_test_repeat, dummy_repeat_probability
    ))
    recency_baseline_pr_auc = float(average_precision_score(
        y_test_repeat, -X_test["recency_days"].fillna(X_test["recency_days"].max()).to_numpy()
    ))
    frequency_baseline_pr_auc = float(average_precision_score(
        y_test_repeat, X_test["orders_180d"].fillna(0).to_numpy()
    ))
    lifetime_baseline_pr_auc = float(average_precision_score(
        y_test_repeat, X_test["lifetime_delivered_orders"].fillna(0).to_numpy()
    ))
    test_metrics["dummy_baseline_pr_auc"] = dummy_pr_auc
    test_metrics["recency_baseline_pr_auc"] = recency_baseline_pr_auc
    test_metrics["frequency_baseline_pr_auc"] = frequency_baseline_pr_auc
    test_metrics["lifetime_frequency_baseline_pr_auc"] = lifetime_baseline_pr_auc
    test_metrics["pr_auc_lift_over_dummy"] = float(
        test_metrics["pr_auc"] - dummy_pr_auc
    )

    test_dates = df.iloc[test_idx]["snapshot_date"].reset_index(drop=True)
    test_customer_ids = df.iloc[test_idx]["customer_unique_id"].reset_index(drop=True)
    capacity_metrics = {
        f"{int(capacity * 100)}%": _monthly_capacity_metrics(
            y_test_repeat, test_repeat_probability, test_dates, capacity
        )
        for capacity in (0.05, 0.10, 0.20)
    }
    baseline_repeat_scores = {
        "recency_only": -X_test["recency_days"].fillna(
            X_test["recency_days"].max()
        ).to_numpy(),
        "orders_180d_only": X_test["orders_180d"].fillna(0).to_numpy(),
        "lifetime_frequency_only": X_test[
            "lifetime_delivered_orders"
        ].fillna(0).to_numpy(),
    }
    baseline_capacity_metrics = {
        baseline_name: {
            f"{int(capacity * 100)}%": _monthly_capacity_metrics(
                y_test_repeat, scores, test_dates, capacity
            )
            for capacity in (0.05, 0.10, 0.20)
        }
        for baseline_name, scores in baseline_repeat_scores.items()
    }
    cluster_bootstrap_ci = _cluster_bootstrap_pr_auc_ci(
        y_test_repeat,
        test_repeat_probability,
        test_customer_ids,
        n_bootstrap=200,
    )

    train_customer_ids = set(
        df.iloc[train_idx]["customer_unique_id"].astype(str).unique()
    )
    validation_customer_ids = set(
        df.iloc[validation_idx]["customer_unique_id"].astype(str).unique()
    )
    development_customer_ids = train_customer_ids | validation_customer_ids
    unseen_mask = ~test_customer_ids.astype(str).isin(development_customer_ids).to_numpy()
    unseen_metrics: dict[str, Any] = {
        "rows": int(unseen_mask.sum()),
        "customers": int(test_customer_ids[unseen_mask].nunique()),
    }
    if unseen_mask.any():
        unseen_metrics.update(_ranking_summary(
            y_test_repeat[unseen_mask],
            test_repeat_probability[unseen_mask],
        ))
        if len(np.unique(y_test_repeat[unseen_mask])) == 2:
            unseen_metrics["precision_at_10pct_capacity"] = _monthly_capacity_metrics(
                y_test_repeat[unseen_mask],
                test_repeat_probability[unseen_mask],
                test_dates[unseen_mask],
                0.10,
            )["precision"]
        else:
            unseen_metrics["precision_at_10pct_capacity"] = None

    predictions = pd.DataFrame({
        "customer_unique_id": test_customer_ids.to_numpy(),
        "snapshot_date": test_dates.to_numpy(),
        "actual_churned": y_test.to_numpy(),
        "churn_probability": test_churn_probability,
        "repeat_purchase_probability": test_repeat_probability,
        "predicted_churned": test_pred_churn,
        "predicted_repeat_purchase": test_pred_repeat,
        "seen_in_training_or_validation": ~unseen_mask,
        "threshold": threshold,
    })
    predictions.to_parquet(MODEL_DIR / "test_predictions.parquet", index=False)

    precision, recall, pr_thresholds = precision_recall_curve(
        y_test_repeat, test_repeat_probability
    )
    pd.DataFrame({
        "precision": precision[:-1],
        "recall": recall[:-1],
        "repeat_purchase_probability_threshold": pr_thresholds,
    }).to_csv(MODEL_DIR / "test_precision_recall_curve.csv", index=False)

    calibration_bins = _calibration_bins(
        y_test_repeat, test_repeat_probability, n_bins=10
    )
    calibration_bins_path = MODEL_DIR / "test_calibration_bins.csv"
    calibration_bins.to_csv(calibration_bins_path, index=False)

    capacity_path = MODEL_DIR / "test_capacity_metrics.json"
    capacity_path.write_text(
        json.dumps(capacity_metrics, indent=2), encoding="utf-8"
    )
    baseline_capacity_path = MODEL_DIR / "test_baseline_capacity_metrics.json"
    baseline_capacity_path.write_text(
        json.dumps(baseline_capacity_metrics, indent=2), encoding="utf-8"
    )

    report_path = MODEL_DIR / "test_classification_report.json"
    report_path.write_text(json.dumps(classification_report(
        y_test,
        test_pred_churn,
        labels=[0, 1],
        target_names=["second_purchase_within_90_days", "no_purchase_within_90_days"],
        output_dict=True,
        zero_division=0,
    ), indent=2), encoding="utf-8")

    matrix_path = MODEL_DIR / "test_confusion_matrix.json"
    matrix_path.write_text(json.dumps({
        "labels": ["second_purchase_within_90_days", "no_purchase_within_90_days"],
        "matrix": confusion_matrix(y_test, test_pred_churn, labels=[0, 1]).tolist(),
        "threshold": threshold,
    }, indent=2), encoding="utf-8")

    comparison["selected"] = comparison["model"].eq(selected_name)
    comparison_path = MODEL_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    model_path = MODEL_DIR / "best_churn_model.joblib"
    joblib.dump(calibrated_model, model_path)

    metadata = {
        "pipeline_status": "ready",
        "model_version": MODEL_VERSION,
        "cohort_definition": (
            "Eligible after at least one delivered order; predict the second "
            "delivered purchase within the next 90 days."
        ),
        "experiment_name": experiment,
        "tracking_uri": get_tracking_uri(),
        "selected_model": selected_name,
        "selection_metric": "cv_average_precision_for_second_purchase_within_90_days_against_rfm_baselines",
        "calibration": {
            "method": "Platt sigmoid",
            "fit_source": "purged out-of-fold predictions from training partition only",
            "training_oof_rows": calibration_rows,
            "slope": float(calibrator.coef_[0][0]),
            "intercept": float(calibrator.intercept_[0]),
        },
        "threshold_objective": "F2",
        "threshold_beta": 2.0,
        "best_params": selected_search.best_params_,
        "threshold": threshold,
        "threshold_selection": "maximum F2 for second_purchase_within_90_days on separate validation period",
        "repeat_purchase_probability_threshold": repeat_threshold,
        "evaluation_positive_class": "second_purchase_within_90_days (SQL churned=0)",
        "threshold_validation_metrics": validation_threshold_metrics,
        "target_definition": (
            "1 when no eventually-delivered order has a purchase timestamp in "
            "[snapshot_date, snapshot_date + 90 days); otherwise 0. The positive "
            "event is the customer's second delivered purchase after cohort entry."
        ),
        "feature_columns": FEATURE_COLS,
        "feature_availability": {
            "historical_orders": (
                "order_purchase_timestamp < snapshot_date and "
                "order_delivered_customer_date < snapshot_date"
            ),
            "review_score": "review_answer_timestamp < snapshot_date",
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
            "Known-customer future prediction: unique snapshot dates are split "
            "chronologically into train, validation and final test with a 90-day "
            "embargo at period boundaries; CV folds are grouped by unique date and "
            "purge overlapping 90-day label windows. A separate unseen-customer "
            "subset is also reported for customers absent from both train and "
            "validation partitions."
        ),
        "periods": periods,
        "customer_overlap": {
            "train_customers": int(len(train_customer_ids)),
            "validation_customers": int(len(validation_customer_ids)),
            "test_customers": int(test_customer_ids.nunique()),
            "test_customers_seen_in_development": int(
                len(set(test_customer_ids.astype(str)) & development_customer_ids)
            ),
            "test_customers_unseen_in_development": int(
                test_customer_ids[unseen_mask].nunique()
            ),
        },
        "class_distribution": {
            "all": _distribution(y),
            "train": _distribution(y_train),
            "validation": _distribution(y_validation),
            "test": _distribution(y_test),
        },
        "test_metrics": test_metrics,
        "test_capacity_metrics": capacity_metrics,
        "baseline_capacity_metrics": baseline_capacity_metrics,
        "customer_cluster_bootstrap_pr_auc_ci95": cluster_bootstrap_ci,
        "unseen_customer_test_metrics": unseen_metrics,
        "baseline_metrics": {
            "prior_dummy_pr_auc": dummy_pr_auc,
            "recency_only_pr_auc": recency_baseline_pr_auc,
            "orders_180d_only_pr_auc": frequency_baseline_pr_auc,
            "lifetime_orders_only_pr_auc": lifetime_baseline_pr_auc,
        },
        "dummy_baseline": {
            "strategy": "prior",
            "test_pr_auc": dummy_pr_auc,
            "test_positive_prevalence": float(y_test_repeat.mean()),
        },
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "artifacts": {
            "model": "models/best_churn_model.joblib",
            "comparison": "models/model_comparison.csv",
            "test_predictions": "models/test_predictions.parquet",
            "classification_report": "models/test_classification_report.json",
            "confusion_matrix": "models/test_confusion_matrix.json",
            "precision_recall_curve": "models/test_precision_recall_curve.csv",
            "capacity_metrics": "models/test_capacity_metrics.json",
            "baseline_capacity_metrics": "models/test_baseline_capacity_metrics.json",
            "calibration_bins": "models/test_calibration_bins.csv",
        },
    }
    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    with mlflow.start_run(run_name="final_selected_model_holdout"):
        mlflow.log_params({
            "selected_model": selected_name,
            "selection_metric": "cv_average_precision_for_second_purchase_within_90_days_against_rfm_baselines",
            "model_version": MODEL_VERSION,
            "threshold": threshold,
            "threshold_selection": "maximum validation F2 for second_purchase_within_90_days",
            "threshold_beta": 2.0,
            "calibration_method": "Platt sigmoid fitted on purged OOF training predictions",
            "test_start": periods["test_start"],
            "test_end": periods["test_end"],
        })
        mlflow.log_metrics({
            f"test_{key}": float(value)
            for key, value in test_metrics.items()
            if isinstance(value, (float, int)) and np.isfinite(value)
        })
        mlflow.log_metric("validation_repeat_purchase_f1", validation_threshold_metrics["f1"])
        mlflow.log_metric("validation_repeat_purchase_f2", validation_threshold_metrics["f2"])
        mlflow.log_metric("validation_repeat_purchase_precision", validation_threshold_metrics["precision"])
        mlflow.log_metric("validation_repeat_purchase_recall", validation_threshold_metrics["recall"])
        mlflow.log_metric("customer_cluster_pr_auc_ci95_lower", cluster_bootstrap_ci["lower_95"])
        mlflow.log_metric("customer_cluster_pr_auc_ci95_upper", cluster_bootstrap_ci["upper_95"])
        for artifact, folder in [
            (comparison_path, "model_selection"),
            (metadata_path, "model_selection"),
            (report_path, "evaluation"),
            (matrix_path, "evaluation"),
            (MODEL_DIR / "test_precision_recall_curve.csv", "evaluation"),
            (MODEL_DIR / "test_predictions.parquet", "evaluation"),
            (capacity_path, "evaluation"),
            (baseline_capacity_path, "evaluation"),
            (calibration_bins_path, "evaluation"),
        ]:
            mlflow.log_artifact(str(artifact), artifact_path=folder)
        mlflow.sklearn.log_model(
            calibrated_model, artifact_path="best_churn_model", input_example=X_train.head(3)
        )

    print("Target distribution (1 = no second purchase; 0 = second purchase within 90 days):")
    print(y.value_counts(normalize=True).sort_index().to_string())
    print("Periods:", json.dumps(periods, indent=2))
    print("CV model comparison (PR-AUC for second purchase within 90 days):")
    print(comparison.to_string(index=False))
    print("Selected model:", selected_name)
    print("Validation-selected threshold objective: F2 (beta=2.0)")
    print("Validation-selected churn probability threshold:", threshold)
    print("Validation-selected repeat-purchase probability threshold:", repeat_threshold)
    print("Final untouched test metrics:", json.dumps(test_metrics, indent=2))
    print("Monthly contact-capacity metrics:", json.dumps(capacity_metrics, indent=2))
    print("Customer-clustered PR-AUC 95% CI:", json.dumps(cluster_bootstrap_ci, indent=2))
    print("Unseen-customer test results:", json.dumps(unseen_metrics, indent=2))


if __name__ == "__main__":
    main()
