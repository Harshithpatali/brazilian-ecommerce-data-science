from src.features.churn_features import (
    CATEGORICAL_FEATURE_COLS,
    FEATURE_COLS,
    NUMERIC_FEATURE_COLS,
)


def test_numeric_contract_keeps_purchase_history_and_value_features():
    expected = {
        "orders_180d", "orders_90d", "orders_30d", "orders_prior_180d",
        "lifetime_delivered_orders", "customer_tenure_days",
        "positive_interpurchase_gap_count", "zero_interpurchase_gap_share",
        "avg_interpurchase_days", "median_interpurchase_days",
        "interpurchase_gap_cv", "recency_to_avg_interpurchase_ratio",
        "recent_vs_prior_order_frequency", "revenue_180d", "revenue_90d",
        "revenue_30d", "revenue_prior_180d", "avg_order_value_180d",
        "avg_review_score_180d", "avg_delivery_delay_180d",
        "active_months_180d", "recency_days", "recent_order_share",
        "recent_revenue_share", "orders_per_active_month",
        "revenue_per_order", "recent_vs_older_order_ratio",
    }
    assert expected == set(NUMERIC_FEATURE_COLS)


def test_categorical_contract_includes_customer_state_and_snapshot_seasonality():
    assert CATEGORICAL_FEATURE_COLS == ["customer_state", "snapshot_month"]


def test_v5_feature_contract_is_unique_and_has_29_columns():
    assert len(NUMERIC_FEATURE_COLS) == 27
    assert len(CATEGORICAL_FEATURE_COLS) == 2
    assert len(FEATURE_COLS) == 29
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))


def test_categorical_feature_normalization_is_deterministic():
    import pandas as pd

    from src.features.churn_features import add_churn_features

    source = pd.DataFrame({
        "customer_state": ["sp", None, "RJ"],
        "snapshot_month": [1, None, 12],
    })
    result = add_churn_features(source)

    assert result["customer_state"].tolist() == ["SP", "UNKNOWN", "RJ"]
    assert result["snapshot_month"].tolist() == ["1", "-1", "12"]
