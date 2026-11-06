from scripts.check_source_schema import (
    REQUIRED_SOURCE_COLUMNS,
    find_missing_source_columns,
)


def all_required_columns():
    return {
        (table, column)
        for table, columns in REQUIRED_SOURCE_COLUMNS.items()
        for column in columns
    }


def test_complete_olist_source_schema_passes_preflight():
    assert find_missing_source_columns(all_required_columns()) == {}


def test_missing_review_answer_timestamp_is_detected_before_build():
    available = all_required_columns()
    available.remove(("olist_order_reviews", "review_answer_timestamp"))
    missing = find_missing_source_columns(available)
    assert missing == {
        "olist_order_reviews": ["review_answer_timestamp"]
    }


def test_missing_raw_source_table_columns_are_reported_by_table():
    available = all_required_columns()
    available = {
        item for item in available
        if item[0] != "olist_order_payments"
    }
    missing = find_missing_source_columns(available)
    assert set(missing["olist_order_payments"]) == {
        "order_id", "payment_value"
    }


def test_build_all_runs_schema_preflight_before_sql_folders():
    from pathlib import Path

    build_all = (
        Path(__file__).resolve().parents[1] / "scripts" / "build_all.py"
    ).read_text(encoding="utf-8")
    assert build_all.index("scripts.check_source_schema") < build_all.index("for folder in FOLDERS")
