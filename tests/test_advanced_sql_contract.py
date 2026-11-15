from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SQL_PATH = ROOT / "sql" / "17_advanced_customer_science" / "01_first_order_180d_target.sql"
BUILD_PATH = ROOT / "scripts" / "build_all.py"
VALIDATOR_PATH = ROOT / "scripts" / "validate_advanced_customer_science.py"


def test_advanced_target_is_in_reproducible_build():
    build = BUILD_PATH.read_text(encoding="utf-8")
    assert '"sql/17_advanced_customer_science"' in build
    assert "scripts.validate_advanced_customer_science" in build


def test_target_uses_delivered_rank_and_full_maturity_window():
    sql = SQL_PATH.read_text(encoding="utf-8")
    assert "ROW_NUMBER() OVER(PARTITION BY customer_unique_id" in sql
    assert "order_status='delivered'" in sql
    assert "WHERE rn=1" in sql and "WHERE rn=2" in sql
    assert "INTERVAL '180 days'<=c.max_purchase" in sql


def test_database_validator_checks_grain_maturity_and_label_consistency():
    validator = VALIDATOR_PATH.read_text(encoding="utf-8")
    for token in ("COUNT(DISTINCT customer_unique_id)", "null_customer_ids", "immature_rows",
                  "inconsistent_positive_labels", "inconsistent_negative_labels",
                  "to_regclass('research.first_order_repeat_180d')"):
        assert token in validator
