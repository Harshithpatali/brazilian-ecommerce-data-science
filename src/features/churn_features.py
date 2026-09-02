from __future__ import annotations

import pandas as pd


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


def add_churn_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return the SQL-generated, leakage-safe churn feature contract."""
    return df.copy()
