from __future__ import annotations

import pandas as pd


NUMERIC_FEATURE_COLS = [
    "orders_180d",
    "orders_90d",
    "orders_30d",
    "orders_prior_180d",
    "lifetime_delivered_orders",
    "customer_tenure_days",
    "positive_interpurchase_gap_count",
    "zero_interpurchase_gap_share",
    "avg_interpurchase_days",
    "median_interpurchase_days",
    "interpurchase_gap_cv",
    "recency_to_avg_interpurchase_ratio",
    "recent_vs_prior_order_frequency",
    "revenue_180d",
    "revenue_90d",
    "revenue_30d",
    "revenue_prior_180d",
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

CATEGORICAL_FEATURE_COLS = [
    "customer_state",
    "snapshot_month",
]

FEATURE_COLS = [*NUMERIC_FEATURE_COLS, *CATEGORICAL_FEATURE_COLS]


def add_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize categorical fields without deriving information from future data."""
    result = df.copy()
    if "customer_state" in result.columns:
        result["customer_state"] = (
            result["customer_state"].fillna("UNKNOWN").astype(str).str.upper()
        )
    if "snapshot_month" in result.columns:
        result["snapshot_month"] = (
            pd.to_numeric(result["snapshot_month"], errors="coerce")
            .fillna(-1)
            .astype("int64")
            .astype(str)
        )
    return result
