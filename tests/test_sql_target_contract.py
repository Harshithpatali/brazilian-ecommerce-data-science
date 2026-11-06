from pathlib import Path


SQL_PATH = (
    Path(__file__).resolve().parents[1]
    / "sql"
    / "14_ml_features"
    / "01_customer_churn_dataset.sql"
)


def test_sql_uses_half_open_90_day_target_window():
    sql = SQL_PATH.read_text(encoding="utf-8")
    assert "future_order.order_purchase_timestamp >= s.snapshot_date" in sql
    assert "future_order.order_purchase_timestamp" in sql
    assert "< s.snapshot_date + INTERVAL '90 days'" in sql


def test_sql_features_require_delivery_before_snapshot():
    sql = SQL_PATH.read_text(encoding="utf-8")
    assert "d.order_delivered_customer_date < s.snapshot_date" in sql
    assert "r.review_answer_timestamp < s.snapshot_date" in sql


def test_sql_requires_full_future_observation_window():
    sql = SQL_PATH.read_text(encoding="utf-8")
    assert "generated.snapshot_timestamp + INTERVAL '90 days'" in sql
    assert "<= o.max_observed_purchase_timestamp" in sql



def test_review_score_uses_answer_timestamp_and_order_weighting():
    sql = SQL_PATH.read_text(encoding="utf-8")
    assert "r.review_answer_timestamp < s.snapshot_date" in sql
    assert "r.review_creation_date < s.snapshot_date" not in sql
    assert "AVG(order_review.order_avg_review_score)" in sql



def test_history_features_are_restricted_to_orders_observable_before_snapshot():
    sql = SQL_PATH.read_text(encoding="utf-8")
    assert "d.order_purchase_timestamp < s.snapshot_date" in sql
    assert "d.order_delivered_customer_date < s.snapshot_date" in sql
    assert "LAG(d.order_purchase_timestamp)" in sql
    assert "lifetime_delivered_orders" in sql
    assert "median_interpurchase_days" in sql
    assert "positive_interpurchase_gap_count" in sql
    assert "zero_interpurchase_gap_share" in sql
    assert "FILTER (WHERE history.gap_days > 0)" in sql
    assert "orders_prior_180d" in sql
    assert "s.snapshot_date - INTERVAL '360 days'" in sql
    assert "s.snapshot_date - INTERVAL '180 days'" in sql



def test_v5_cohort_starts_after_first_delivered_order_to_predict_second_purchase():
    sql = SQL_PATH.read_text(encoding="utf-8")
    assert "first_delivered AS" in sql
    assert "WHERE delivered_order_number = 1" in sql
    assert "second_delivered" not in sql
    assert "after the first delivered order" in sql.lower()
