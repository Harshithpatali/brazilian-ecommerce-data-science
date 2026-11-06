from src.features.churn_features import FEATURE_COLS


def test_ml_feature_contract_includes_recent_and_prior_activity():
    assert "orders_180d" in FEATURE_COLS
    assert "orders_90d" in FEATURE_COLS
    assert "orders_30d" in FEATURE_COLS
    assert "orders_prior_180d" in FEATURE_COLS
    assert "recent_vs_prior_order_frequency" in FEATURE_COLS


def test_ml_feature_contract_includes_lifetime_purchase_behavior():
    assert "lifetime_delivered_orders" in FEATURE_COLS
    assert "customer_tenure_days" in FEATURE_COLS
    assert "avg_interpurchase_days" in FEATURE_COLS
    assert "median_interpurchase_days" in FEATURE_COLS
    assert "interpurchase_gap_cv" in FEATURE_COLS
    assert "recency_to_avg_interpurchase_ratio" in FEATURE_COLS


def test_ml_feature_contract_is_unique_and_contains_all_expected_features():
    assert len(FEATURE_COLS) == 25
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
    assert "revenue_prior_180d" in FEATURE_COLS
    assert "avg_review_score_180d" in FEATURE_COLS
