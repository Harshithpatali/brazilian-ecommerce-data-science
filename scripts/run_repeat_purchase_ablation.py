"""Training-only temporal feature ablation for second-purchase prediction.

This diagnostic deliberately never scores the validation or final test partitions.
It answers whether feature families add signal beyond lifetime order frequency
inside purged expanding-window CV. It cannot certify production eligibility.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import GridSearchCV

from src.features.churn_features import FEATURE_COLS
from src.models.train_churn import (
    DATA,
    build_models,
    chronological_partition_indices,
    make_date_grouped_splits,
    repeat_purchase_average_precision_scorer,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "model_research" / "repeat_purchase_ablation_cv.csv"
ANCHOR = "lifetime_delivered_orders"


def feature_groups() -> dict[str, list[str]]:
    """Return interpretable groups; every group retains the lifetime baseline."""
    prefixes = {
        "first_order_profile": ("first_order_",),
        "recency_frequency": (
            "orders_", "recency_", "customer_tenure", "interpurchase_",
            "recent_vs_", "active_months_", "orders_per_active_month",
        ),
        "transaction_diversity": (
            "avg_items_", "avg_categories_", "avg_sellers_",
            "avg_distinct_products_", "avg_product_diversity_",
            "category_preference_", "dominant_category_",
            "seller_concentration_", "top_seller_", "payment_type_",
            "installment_", "credit_card_", "avg_payment_",
        ),
        "service_review": (
            "avg_review_", "avg_delivery_", "first_order_delivery_",
            "first_order_approval_", "first_order_review_",
        ),
        "revenue_value": ("revenue_", "avg_order_value_", "revenue_per_order"),
    }
    groups = {"lifetime_frequency_only": [ANCHOR]}
    for name, group_prefixes in prefixes.items():
        columns = [c for c in FEATURE_COLS if c.startswith(group_prefixes)]
        groups[f"{name}_plus_lifetime"] = sorted(set([ANCHOR, *columns]))
    groups["full_feature_set"] = list(FEATURE_COLS)
    return groups


def run(model_names: list[str]) -> pd.DataFrame:
    if not DATA.exists():
        raise SystemExit(
            "Missing data/ml/customer_churn_dataset.parquet. Rebuild the SQL "
            "feature table before running this experiment."
        )
    frame = pd.read_parquet(DATA)
    required = {"snapshot_date", "churned", *FEATURE_COLS}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SystemExit(f"Missing required columns: {missing}")
    frame = frame.dropna(subset=["snapshot_date", "churned"]).copy()
    frame["snapshot_date"] = pd.to_datetime(frame["snapshot_date"], errors="coerce")
    frame = frame.dropna(subset=["snapshot_date"]).sort_values("snapshot_date")
    frame["churned"] = frame["churned"].astype(int)

    train_idx, _, _, periods = chronological_partition_indices(frame["snapshot_date"])
    train = frame.iloc[train_idx].reset_index(drop=True)
    y = train["churned"]
    dates = train["snapshot_date"]
    folds = make_date_grouped_splits(
        dates, y=y, n_splits=min(4, dates.dt.normalize().nunique() - 5),
        min_train_repeat_events=10, min_validation_repeat_events=5,
    )
    available_models = build_models()
    rows: list[dict[str, object]] = []
    for group_name, columns in feature_groups().items():
        X = train[columns].copy()
        candidates = {
            name: available_models[name]
            for name in model_names + ["baseline_lifetime_frequency_only"]
            if name in available_models
        }
        for model_name, (estimator, grid) in candidates.items():
            search = GridSearchCV(
                estimator, grid,
                scoring=repeat_purchase_average_precision_scorer,
                cv=folds, n_jobs=2, pre_dispatch=2,
                refit=False, error_score="raise",
            )
            search.fit(X, y)
            rows.append({
                "feature_group": group_name,
                "feature_count": len(columns),
                "model": model_name,
                "purged_temporal_cv_pr_auc": float(search.best_score_),
                "best_params": str(search.best_params_),
                "train_start": periods["train_start"],
                "train_end": periods["train_end"],
                "validation_and_test_used": False,
            })
    result = pd.DataFrame(rows).sort_values(
        ["feature_group", "purged_temporal_cv_pr_auc"],
        ascending=[True, False],
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(result.to_string(index=False))
    print(f"Saved training-only results to {OUTPUT}")
    print(
        "NOTE: This is feature diagnosis on training-only temporal CV. "
        "Do not use it to claim a holdout win or production eligibility."
    )
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+",
        default=["logistic_regression", "hist_gradient_boosting"],
        help="Candidate model families from src.models.train_churn.build_models",
    )
    args = parser.parse_args()
    run(args.models)
