from src.models.train_churn import FEATURE_COLS

def test_ml_feature_contract():
    assert "recency_days" in FEATURE_COLS
    assert "orders_180d" in FEATURE_COLS
    assert "revenue_180d" in FEATURE_COLS
    assert len(FEATURE_COLS) >= 8
