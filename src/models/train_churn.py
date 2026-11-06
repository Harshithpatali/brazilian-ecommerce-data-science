from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, classification_report, confusion_matrix,
    f1_score, precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .mlflow_utils import configure_mlflow, get_tracking_uri

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ml" / "customer_churn_dataset.parquet"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

MODEL_VERSION = "churn-pit-v2"
FEATURE_COLS = [
    "orders_180d", "orders_90d", "orders_30d",
    "revenue_180d", "revenue_90d", "revenue_30d",
    "avg_order_value_180d", "avg_review_score_180d",
    "avg_delivery_delay_180d", "active_months_180d",
    "recency_days", "recent_order_share", "recent_revenue_share",
    "orders_per_active_month", "revenue_per_order",
    "recent_vs_older_order_ratio",
]


def _dates(values: pd.Series | list[Any]) -> pd.Series:
    result = pd.to_datetime(pd.Series(values), errors="coerce").dt.normalize()
    if result.isna().any():
        raise ValueError("snapshot_date contains invalid or missing values.")
    return result.reset_index(drop=True)


def chronological_partition_indices(
    snapshot_dates: pd.Series | list[Any],
    test_fraction: float = 0.20,
    validation_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, str]]:
    """Partition whole dates into approximately 64%/16%/20% periods."""
    if not 0 < test_fraction < 0.5 or not 0 < validation_fraction < 0.5:
        raise ValueError("Partition fractions must be between 0 and 0.5.")
    row_dates = _dates(snapshot_dates)
    unique = np.array(sorted(row_dates.unique()))
    if len(unique) < 10:
        raise ValueError("At least 10 unique snapshot dates are required for train, validation, CV, and test.")

    test_start = int(np.floor(len(unique) * (1 - test_fraction)))
    test_start = min(max(test_start, 2), len(unique) - 1)
    validation_count = max(1, int(np.floor(test_start * validation_fraction)))
    validation_start = test_start - validation_count
    if validation_start < 6:
        raise ValueError("Too few training dates remain for date-grouped temporal CV.")

    train_dates = unique[:validation_start]
    validation_dates = unique[validation_start:test_start]
    test_dates = unique[test_start:]
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
    }
    return *indices, periods


def make_date_grouped_splits(
    snapshot_dates: pd.Series | list[Any],
    y: pd.Series | np.ndarray | None = None,
    n_splits: int = 4,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Build expanding-window folds over unique dates.

    When labels are supplied, shorten the fold count or extend the initial
    training window until every training and validation fold contains both
    classes. This avoids misleading AP scores from single-class time blocks.
    """
    row_dates = _dates(snapshot_dates)
    unique = np.array(sorted(row_dates.unique()))
    if n_splits < 2 or len(unique) < n_splits + 4:
        raise ValueError("Not enough unique dates for temporal CV.")

    values = row_dates.to_numpy()
    labels = None if y is None else np.asarray(pd.Series(y).reset_index(drop=True))
    if labels is not None and len(labels) != len(values):
        raise ValueError("y and snapshot_dates must have the same number of rows.")

    if labels is None:
        splitter = TimeSeriesSplit(n_splits=n_splits)
        folds = []
        for train_date_idx, valid_date_idx in splitter.split(unique):
            train_dates = unique[train_date_idx]
            valid_dates = unique[valid_date_idx]
            folds.append((
                np.flatnonzero(np.isin(values, train_dates)),
                np.flatnonzero(np.isin(values, valid_dates)),
            ))
        return folds

    minimum_warmup = max(4, int(np.ceil(len(unique) * 0.25)))
    for fold_count in range(min(n_splits, len(unique) - minimum_warmup), 1, -1):
        for warmup_end in range(minimum_warmup, len(unique) - fold_count + 1):
            remaining_dates = unique[warmup_end:]
            blocks = [block for block in np.array_split(remaining_dates, fold_count) if len(block)]
            if len(blocks) != fold_count:
                continue

            candidate_folds = []
            valid_candidate = True
            for block in blocks:
                training_dates = unique[unique < block[0]]
                training_idx = np.flatnonzero(np.isin(values, training_dates))
                validation_idx = np.flatnonzero(np.isin(values, block))
                if len(np.unique(labels[training_idx])) != 2:
                    valid_candidate = False
                    break
                if len(np.unique(labels[validation_idx])) != 2:
                    valid_candidate = False
                    break
                candidate_folds.append((training_idx, validation_idx))

            if valid_candidate:
                return candidate_folds

    raise ValueError(
        "Could not create at least two chronological CV folds containing both "
        "classes in training and validation. Inspect class prevalence by month "
        "or collect a longer observation period."
    )


def choose_f1_threshold(
    y_true: pd.Series | np.ndarray, probabilities: np.ndarray
) -> tuple[float, dict[str, float]]:
    """Choose a threshold on validation data only."""
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    if not len(thresholds):
        threshold = 0.5
    else:
        denom = precision[:-1] + recall[:-1]
        f1_values = np.divide(
            2 * precision[:-1] * recall[:-1], denom,
            out=np.zeros_like(denom), where=denom > 0,
        )
        threshold = float(thresholds[int(np.nanargmax(f1_values))])
    predicted = (probabilities >= threshold).astype(int)
    return threshold, {
        "precision": float(precision_score(y_true, predicted, zero_division=0)),
        "recall": float(recall_score(y_true, predicted, zero_division=0)),
        "f1": float(f1_score(y_true, predicted, zero_division=0)),
    }


def build_models() -> dict[str, tuple[Pipeline, dict[str, list[Any]]]]:
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
                "model__learning_rate": [0.03, 0.08],
                "model__max_leaf_nodes": [15, 31],
                "model__min_samples_leaf": [20, 50],
                "model__l2_regularization": [0.0, 1.0],
            },
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

    train_idx, validation_idx, test_idx, periods = chronological_partition_indices(
        df["snapshot_date"]
    )
    X_train, y_train = X.iloc[train_idx].reset_index(drop=True), y.iloc[train_idx].reset_index(drop=True)
    X_validation = X.iloc[validation_idx].reset_index(drop=True)
    y_validation = y.iloc[validation_idx].reset_index(drop=True)
    X_test, y_test = X.iloc[test_idx].reset_index(drop=True), y.iloc[test_idx].reset_index(drop=True)

    _require_both_classes(y_train, "Training partition")
    _require_both_classes(y_validation, "Validation partition")
    _require_both_classes(y_test, "Final test partition")

    train_dates = df["snapshot_date"].iloc[train_idx].reset_index(drop=True)
    n_dates = int(train_dates.dt.normalize().nunique())
    folds = make_date_grouped_splits(train_dates, y=y_train, n_splits=min(4, n_dates - 4))

    experiment = configure_mlflow()
    searches: dict[str, GridSearchCV] = {}
    rows: list[dict[str, Any]] = []

    for name, (pipeline, grid) in build_models().items():
        with mlflow.start_run(run_name=f"cv_tuned_{name}"):
            search = GridSearchCV(
                pipeline, grid, scoring="average_precision", cv=folds,
                n_jobs=-1, refit=True, error_score="raise",
            )
            search.fit(X_train, y_train)
            searches[name] = search
            mlflow.log_params({
                "model_family": name,
                "cv": "expanding_window_grouped_by_snapshot_date",
                "cv_folds": len(folds),
                "selection_metric": "cv_average_precision",
                "model_version": MODEL_VERSION,
            })
            mlflow.log_params({key: str(value) for key, value in search.best_params_.items()})
            mlflow.log_metric("cv_pr_auc", float(search.best_score_))
            rows.append({
                "model": name,
                "cv_pr_auc": float(search.best_score_),
                "best_params": json.dumps(search.best_params_, sort_keys=True),
            })

    # The winning candidate is selected only by temporal CV, never by test metrics.
    comparison = pd.DataFrame(rows).sort_values(
        ["cv_pr_auc", "model"], ascending=[False, True]
    ).reset_index(drop=True)
    selected_name = str(comparison.iloc[0]["model"])
    selected_search = searches[selected_name]
    selected_model = selected_search.best_estimator_

    # Validation selects the threshold; the final chronological test stays untouched.
    validation_prob = selected_model.predict_proba(X_validation)[:, 1]
    threshold, validation_threshold_metrics = choose_f1_threshold(
        y_validation, validation_prob
    )
    test_prob = selected_model.predict_proba(X_test)[:, 1]
    test_pred = (test_prob >= threshold).astype(int)

    test_metrics = {
        "roc_auc": float(roc_auc_score(y_test, test_prob)),
        "pr_auc": float(average_precision_score(y_test, test_prob)),
        "precision": float(precision_score(y_test, test_pred, zero_division=0)),
        "recall": float(recall_score(y_test, test_pred, zero_division=0)),
        "f1": float(f1_score(y_test, test_pred, zero_division=0)),
        "positive_prevalence": float(y_test.mean()),
        "threshold": float(threshold),
    }
    dummy = DummyClassifier(strategy="prior").fit(X_train, y_train)
    dummy_prob = dummy.predict_proba(X_test)[:, 1]
    dummy_pr_auc = float(average_precision_score(y_test, dummy_prob))
    test_metrics["dummy_baseline_pr_auc"] = dummy_pr_auc
    test_metrics["pr_auc_lift_over_dummy"] = float(test_metrics["pr_auc"] - dummy_pr_auc)

    predictions = pd.DataFrame({
        "customer_unique_id": df.iloc[test_idx]["customer_unique_id"].to_numpy(),
        "snapshot_date": df.iloc[test_idx]["snapshot_date"].to_numpy(),
        "actual_churned": y_test.to_numpy(),
        "churn_probability": test_prob,
        "predicted_churned": test_pred,
        "threshold": threshold,
    })
    predictions.to_parquet(MODEL_DIR / "test_predictions.parquet", index=False)

    precision, recall, pr_thresholds = precision_recall_curve(y_test, test_prob)
    pd.DataFrame({
        "precision": precision[:-1], "recall": recall[:-1], "threshold": pr_thresholds
    }).to_csv(MODEL_DIR / "test_precision_recall_curve.csv", index=False)

    report_path = MODEL_DIR / "test_classification_report.json"
    report_path.write_text(json.dumps(classification_report(
        y_test, test_pred, labels=[0, 1],
        target_names=["purchase_within_90_days", "churned"],
        output_dict=True, zero_division=0,
    ), indent=2), encoding="utf-8")

    matrix_path = MODEL_DIR / "test_confusion_matrix.json"
    matrix_path.write_text(json.dumps({
        "labels": ["purchase_within_90_days", "churned"],
        "matrix": confusion_matrix(y_test, test_pred, labels=[0, 1]).tolist(),
        "threshold": threshold,
    }, indent=2), encoding="utf-8")

    comparison["selected"] = comparison["model"].eq(selected_name)
    comparison_path = MODEL_DIR / "model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    model_path = MODEL_DIR / "best_churn_model.joblib"
    joblib.dump(selected_model, model_path)

    metadata = {
        "pipeline_status": "ready",
        "model_version": MODEL_VERSION,
        "experiment_name": experiment,
        "tracking_uri": get_tracking_uri(),
        "selected_model": selected_name,
        "selection_metric": "cv_pr_auc",
        "best_params": selected_search.best_params_,
        "threshold": threshold,
        "threshold_selection": "maximum F1 on separate validation period",
        "threshold_validation_metrics": validation_threshold_metrics,
        "target_definition": (
            "1 when no eventually-delivered order has a purchase timestamp in "
            "[snapshot_date, snapshot_date + 90 days); otherwise 0."
        ),
        "feature_columns": FEATURE_COLS,
        "feature_window_days": 180,
        "prediction_window_days": 90,
        "evaluation_design": (
            "Unique snapshot dates split chronologically into train, validation "
            "and final test; CV folds grouped by unique date. Customers can appear "
            "in earlier and later periods because the use case is future scoring "
            "of known customers."
        ),
        "periods": periods,
        "class_distribution": {
            "all": _distribution(y),
            "train": _distribution(y_train),
            "validation": _distribution(y_validation),
            "test": _distribution(y_test),
        },
        "test_metrics": test_metrics,
        "dummy_baseline": {
            "strategy": "prior",
            "test_pr_auc": dummy_pr_auc,
            "test_positive_prevalence": float(y_test.mean()),
        },
        "artifacts": {
            "model": "models/best_churn_model.joblib",
            "comparison": "models/model_comparison.csv",
            "test_predictions": "models/test_predictions.parquet",
            "classification_report": "models/test_classification_report.json",
            "confusion_matrix": "models/test_confusion_matrix.json",
            "precision_recall_curve": "models/test_precision_recall_curve.csv",
        },
    }
    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    with mlflow.start_run(run_name="final_selected_model_holdout"):
        mlflow.log_params({
            "selected_model": selected_name,
            "selection_metric": "cv_pr_auc",
            "model_version": MODEL_VERSION,
            "threshold": threshold,
            "threshold_selection": "maximum validation F1",
            "test_start": periods["test_start"],
            "test_end": periods["test_end"],
        })
        mlflow.log_metrics({
            f"test_{key}": float(value)
            for key, value in test_metrics.items()
            if isinstance(value, (float, int)) and np.isfinite(value)
        })
        mlflow.log_metric("validation_threshold_f1", validation_threshold_metrics["f1"])
        for artifact, folder in [
            (comparison_path, "model_selection"),
            (metadata_path, "model_selection"),
            (report_path, "evaluation"),
            (matrix_path, "evaluation"),
        ]:
            mlflow.log_artifact(str(artifact), artifact_path=folder)
        mlflow.sklearn.log_model(
            selected_model, artifact_path="best_churn_model", input_example=X_train.head(3)
        )

    print("Target distribution (1 = churned):")
    print(y.value_counts(normalize=True).sort_index().to_string())
    print("Periods:", json.dumps(periods, indent=2))
    print("CV model comparison:")
    print(comparison.to_string(index=False))
    print("Selected model:", selected_name)
    print("Validation-selected threshold:", threshold)
    print("Final untouched test metrics:", json.dumps(test_metrics, indent=2))


if __name__ == "__main__":
    main()
