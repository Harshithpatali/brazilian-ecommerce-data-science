"""Read-only database integrity checks for the first-order 180-day target."""
from __future__ import annotations

from src.database.io import query_to_dataframe


def main() -> None:
    existence = query_to_dataframe("SELECT to_regclass('research.first_order_repeat_180d') AS target_view")
    if existence.empty or existence.iloc[0]["target_view"] is None:
        raise SystemExit("Missing research.first_order_repeat_180d. Run the project SQL build including sql/17_advanced_customer_science.")

    checks = query_to_dataframe("""
        WITH target AS (SELECT * FROM research.first_order_repeat_180d)
        SELECT
            COUNT(*) AS target_rows,
            COUNT(DISTINCT customer_unique_id) AS distinct_customers,
            COUNT(*) FILTER (WHERE customer_unique_id IS NULL) AS null_customer_ids,
            COUNT(*) FILTER (WHERE first_order_id IS NULL) AS null_first_order_ids,
            COUNT(*) FILTER (WHERE repeat_180d NOT IN (0, 1) OR repeat_180d IS NULL) AS invalid_labels,
            COUNT(*) FILTER (WHERE first_purchase_timestamp + INTERVAL '180 days' >
                (SELECT MAX(order_purchase_timestamp) FROM analytics.order_level)) AS immature_rows,
            COUNT(*) FILTER (WHERE repeat_180d = 1 AND (days_to_second_purchase IS NULL OR
                days_to_second_purchase < 0 OR days_to_second_purchase > 180)) AS inconsistent_positive_labels,
            COUNT(*) FILTER (WHERE repeat_180d = 0 AND days_to_second_purchase BETWEEN 0 AND 180)
                AS inconsistent_negative_labels,
            AVG(repeat_180d::numeric) AS repeat_rate_180d,
            MIN(first_purchase_timestamp) AS earliest_index_order,
            MAX(first_purchase_timestamp) AS latest_index_order
        FROM target
    """)
    row = checks.iloc[0]
    failures = []
    if int(row["target_rows"]) == 0:
        failures.append("target view is empty")
    if int(row["target_rows"]) != int(row["distinct_customers"]):
        failures.append("target grain is not one row per customer")
    for field in ("null_customer_ids", "null_first_order_ids", "invalid_labels", "immature_rows",
                  "inconsistent_positive_labels", "inconsistent_negative_labels"):
        if int(row[field] or 0) > 0:
            failures.append(f"{field}={int(row[field])}")
    print("Advanced customer-science target validation")
    print(checks.to_string(index=False))
    if failures:
        raise SystemExit("Validation failed: " + "; ".join(failures))
    print("PASS: one mature first-delivered-order row per customer; 180-day labels are internally consistent.")
    print("This validates target integrity, not predictive performance or causal identification.")


if __name__ == "__main__":
    main()
