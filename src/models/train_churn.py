from __future__ import annotations

import json
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .mlflow_utils import configure_mlflow, get_tracking_uri

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "ml" / "customer_churn_dataset.parquet"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURE_COLS = [
    "orders_180d",
    "orders_90d",
    "orders_30d",
    "revenue_180d",
    "revenue_90d",
    "revenue_30d",
    "avg_order_value_180d",
    "avg_review_score_180d",
    "avg_delivery_delay_180d",
    "active_months_180d",
    "recency_days",
    "recent_order_share",
    "recent_revenue_share",
    "orders_per_active_month",
    "revenue_per_order",
    "recent_vs_older_order_ratio",
]


def build_models() -> dict[str, tuple[Pipeline, dict]]:
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    prep = ColumnTransformer(
        transformers=[("numeric", numeric_pipe, FEATURE_COLS)]
    )

    return {
        "logistic_regression": (
            Pipeline(
                steps=[
                    ("preprocess", prep),
                    ("model", LogisticRegression(max_iter=3000, class_weight="balanced")),
                ]
            ),
            {
                "model__C": [0.05, 0.2, 1.0, 5.0],
                "model__solver": ["liblinear"],
            },
        ),
        "hist_gradient_boosting": (
            Pipeline(
                steps=[
                    ("preprocess", prep),
                    (
                        "model",
                        HistGradientBoostingClassifier(
                            random_state=42,
                            class_weight="balanced",
                        ),
                    ),
                ]
            ),
            {
                "model__max_iter": [150, 300],
                "model__learning_rate": [0.03, 0.08],
                "model__max_leaf_nodes": [15, 31],
                "model__min_samples_leaf": [20, 50],
                "model__l2_regularization": [0.0, 1.0],
            },
        ),
    }


def evaluate(model, X_test, y_test) -> dict[str, float]:
    prob = model.predict_proba(X_test)[:, 1]
    pred = (prob >= 0.5).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_test, prob)),
        "pr_auc": float(average_precision_score(y_test, prob)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
    }


def main() -> None:
    if not DATA.exists():
        raise SystemExit("Run `python -m scripts.build_ml_data` first.")

    df = pd.read_parquet(DATA)
    df["snapshot_date"] = pd.to_datetime(df["snapshot_date"], errors="coerce")
    df = df.dropna(subset=["snapshot_date"]).sort_values(
        ["snapshot_date", "customer_unique_id"]
    ).reset_index(drop=True)

    y = df["churned"].astype(int)
    X = df[FEATURE_COLS].copy()

    dates = np.array(sorted(df["snapshot_date"].dt.date.unique()))
    if len(dates) < 6:
        raise SystemExit("Not enough snapshot dates for a reliable temporal holdout.")

    # Hold out the latest 20% of snapshot dates. This is intentionally temporal.
    split_idx = max(1, int(len(dates) * 0.8))
    cutoff = dates[split_idx - 1]

    train_mask = df["snapshot_date"].dt.date <= cutoff
    test_mask = df["snapshot_date"].dt.date > cutoff

    X_train, X_test = X.loc[train_mask], X.loc[test_mask]
    y_train, y_test = y.loc[train_mask], y.loc[test_mask]

    if X_test.empty or X_train.empty:
        raise SystemExit("Chronological split produced an empty train/test set.")

    experiment_name = configure_mlflow()

    # Time-aware CV across the training period.
    n_splits = min(4, max(2, len(dates[dates <= cutoff]) // 3))
    cv = TimeSeriesSplit(n_splits=n_splits)

    comparison = []
    best_name = None
    best_search = None
    best_score = -np.inf

    for name, (pipeline, grid) in build_models().items():
        with mlflow.start_run(run_name=f"tuned_{name}", nested=True):
            mlflow.log_params({
                "model_family": name,
                "cv": "TimeSeriesSplit",
                "cv_splits": n_splits,
                "selection_metric": "pr_auc",
                "test_strategy": "latest 20_percent_of_snapshot_dates",
                "feature_count": len(FEATURE_COLS),
                "eligible_population": "customers_with_at_least_2_prior_delivered_orders_and_recent_180d_activity",
                "prediction_horizon_days": 90,
            })

            search = GridSearchCV(
                pipeline,
                grid,
                scoring="average_precision",
                cv=cv,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            search.fit(X_train, y_train)

            metrics = evaluate(search.best_estimator_, X_test, y_test)
            metrics["cv_pr_auc"] = float(search.best_score_)

            mlflow.log_params({k: str(v) for k, v in search.best_params_.items()})
            mlflow.log_metrics(metrics)

            report = classification_report(
                y_test,
                (search.best_estimator_.predict_proba(X_test)[:, 1] >= 0.5).astype(int),
                output_dict=True,
            )
            report_path = MODEL_DIR / f"{name}_classification_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(report_path), artifact_path="evaluation")

            comparison.append({
                "model": name,
                **metrics,
                "best_params": json.dumps(search.best_params_),
            })

            if metrics["pr_auc"] > best_score:
                best_score = metrics["pr_auc"]
                best_name = name
                best_search = search

    if best_search is None:
        raise RuntimeError("No model completed successfully.")

    comparison_df = pd.DataFrame(comparison).sort_values(
        ["pr_auc", "roc_auc"], ascending=False
    )
    comparison_path = MODEL_DIR / "model_comparison.csv"
    comparison_df.to_csv(comparison_path, index=False)

    final_model_path = MODEL_DIR / "best_churn_model.joblib"
    joblib.dump(best_search.best_estimator_, final_model_path)

    metadata = {
        "experiment_name": experiment_name,
        "tracking_uri": get_tracking_uri(),
        "selected_model": best_name,
        "selection_metric": "pr_auc",
        "best_params": best_search.best_params_,
        "holdout_cutoff": str(cutoff),
        "holdout_metrics": comparison_df.iloc[0].to_dict(),
        "feature_columns": FEATURE_COLS,
        "eligible_population": "customers_with_at_least_2_prior_delivered_orders_and_recent_180d_activity",
        "feature_window_days": 180,
        "prediction_window_days": 90,
    }
    metadata_path = MODEL_DIR / "model_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, default=str), encoding="utf-8")

    with mlflow.start_run(run_name="final_selected_model"):
        mlflow.log_params({
            "selected_model": best_name,
            "selection_metric": "pr_auc",
            "feature_count": len(FEATURE_COLS),
            "holdout_cutoff": str(cutoff),
            "feature_window_days": 180,
            "prediction_window_days": 90,
        })
        for key in ["roc_auc", "pr_auc", "precision", "recall", "f1", "cv_pr_auc"]:
            value = comparison_df.iloc[0].get(key)
            if value is not None and pd.notna(value):
                mlflow.log_metric(key, float(value))

        mlflow.log_artifact(str(comparison_path), artifact_path="model_selection")
        mlflow.log_artifact(str(metadata_path), artifact_path="model_selection")

        # Infer signature from the fitted pipeline using a small training example.
        example = X_train.head(3)
        signature = mlflow.models.infer_signature(
            example,
            best_search.best_estimator_.predict_proba(example)[:, 1],
        )
        mlflow.sklearn.log_model(
            best_search.best_estimator_,
            artifact_path="best_churn_model",
            signature=signature,
            input_example=example,
        )

    print("\nTarget distribution:")
    print(y.value_counts(normalize=True).rename("proportion").to_string())

    print("\nModel comparison:")
    print(
        comparison_df[
            ["model", "cv_pr_auc", "pr_auc", "roc_auc", "precision", "recall", "f1"]
        ].to_string(index=False)
    )

    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {final_model_path}")
    print(f"MLflow tracking URI: {get_tracking_uri()}")
    print(f"MLflow experiment: {experiment_name}")


if __name__ == "__main__":
    main()
