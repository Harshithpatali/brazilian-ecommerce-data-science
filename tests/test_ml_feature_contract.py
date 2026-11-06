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
    assert "positive_interpurchase_gap_count" in FEATURE_COLS
    assert "zero_interpurchase_gap_share" in FEATURE_COLS
    assert "avg_interpurchase_days" in FEATURE_COLS
    assert "median_interpurchase_days" in FEATURE_COLS
    assert "interpurchase_gap_cv" in FEATURE_COLS
    assert "recency_to_avg_interpurchase_ratio" in FEATURE_COLS


def test_ml_feature_contract_includes_first_order_and_payment_behaviour():
    expected = {
        "first_order_gross_value",
        "first_order_item_count",
        "first_order_distinct_products",
        "first_order_distinct_categories",
        "first_order_seller_count",
        "first_order_freight_share",
        "first_order_max_installments",
        "first_order_payment_type_count",
        "first_order_is_credit_card",
        "first_order_is_boleto",
        "first_order_is_debit_card",
        "first_order_is_voucher",
        "first_order_delivery_days",
        "first_order_delivery_delay_days",
        "first_order_approval_lag_hours",
        "first_order_review_score",
        "avg_items_per_order_180d",
        "avg_categories_per_order_180d",
        "avg_sellers_per_order_180d",
        "avg_payment_installments_180d",
        "credit_card_order_share_180d",
        "avg_freight_share_180d",
    }
    assert expected <= set(FEATURE_COLS)


def test_ml_feature_contract_is_unique_and_contains_all_expected_features():
    assert len(FEATURE_COLS) == 57
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
    assert "revenue_prior_180d" in FEATURE_COLS
    assert "avg_review_score_180d" in FEATURE_COLS


def test_ml_feature_contract_includes_basket_category_seller_and_payment_mix():
    expected = {
        "avg_distinct_products_per_order_180d",
        "avg_product_diversity_share_180d",
        "category_preference_entropy_180d",
        "dominant_category_share_180d",
        "seller_concentration_hhi_180d",
        "top_seller_share_180d",
        "payment_type_diversity_180d",
        "installment_order_share_180d",
    }
    assert expected <= set(FEATURE_COLS)
    assert len(FEATURE_COLS) == len(set(FEATURE_COLS))
