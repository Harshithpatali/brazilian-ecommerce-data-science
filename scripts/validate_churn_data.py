from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.io import query_to_dataframe


EXPECTED_ZERO_CHECKS = (
    "duplicate_keys",
    "target_mismatches",
    "invalid_feature_rows",
    "incomplete_label_windows",
    "review_feature_mismatches",
    "invalid_history_feature_rows",
)


def assert_zero_integrity_checks(checks: dict[str, object]) -> None:
    """Fail validation if an integrity check is missing or non-zero."""
    missing = [name for name in EXPECTED_ZERO_CHECKS if name not in checks]
    if missing:
        raise SystemExit(f"Validation result is missing checks: {missing}")

    violations = {}
    for name in EXPECTED_ZERO_CHECKS:
        try:
            count = int(checks[name])
        except (TypeError, ValueError) as exc:
            raise SystemExit(
                f"Validation check {name!r} did not return an integer."
            ) from exc
        if count != 0:
            violations[name] = count

    if violations:
        raise SystemExit(f"Churn data validation failed: {violations}")


def main() -> None:
    summary = query_to_dataframe("""
        SELECT COUNT(*) AS rows,
               COUNT(DISTINCT customer_unique_id) AS customers,
               COUNT(DISTINCT snapshot_date) AS snapshots,
               MIN(snapshot_date) AS first_snapshot,
               MAX(snapshot_date) AS last_snapshot,
               AVG(churned::numeric) AS churn_rate,
               SUM((churned = 1)::int) AS churned_rows,
               SUM((churned = 0)::int) AS non_churned_rows
        FROM ml.customer_churn_dataset
    """)
    print("Dataset summary")
    print(summary.to_string(index=False))

    distribution = query_to_dataframe("""
        SELECT churned, COUNT(*) AS rows,
               ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS percentage
        FROM ml.customer_churn_dataset
        GROUP BY churned ORDER BY churned
    """)
    print("\nTarget distribution (1 = no delivered purchase in next 90 days)")
    print(distribution.to_string(index=False))

    monthly = query_to_dataframe("""
        SELECT snapshot_date, COUNT(*) AS rows,
               SUM((churned = 1)::int) AS churned_rows,
               SUM((churned = 0)::int) AS non_churned_rows,
               ROUND(AVG(churned::numeric), 4) AS churn_rate
        FROM ml.customer_churn_dataset
        GROUP BY snapshot_date ORDER BY snapshot_date
    """)
    print("\nMonthly target distribution")
    print(monthly.to_string(index=False))

    checks = query_to_dataframe("""
        WITH source_end AS (
            SELECT MAX(order_purchase_timestamp) AS max_purchase_timestamp
            FROM analytics.order_level
        )
        SELECT
          (SELECT COUNT(*) FROM (
              SELECT customer_unique_id, snapshot_date
              FROM ml.customer_churn_dataset
              GROUP BY customer_unique_id, snapshot_date HAVING COUNT(*) > 1
          ) d) AS duplicate_keys,
          (SELECT COUNT(*) FILTER (
              WHERE dataset.churned <> CASE WHEN EXISTS (
                  SELECT 1 FROM ml.delivered_orders f
                  WHERE f.customer_unique_id = dataset.customer_unique_id
                    AND f.order_purchase_timestamp >= dataset.snapshot_date
                    AND f.order_purchase_timestamp < dataset.snapshot_date + INTERVAL '90 days'
              ) THEN 0 ELSE 1 END
           ) FROM ml.customer_churn_dataset dataset) AS target_mismatches,
          (SELECT COUNT(*) FILTER (
              WHERE orders_180d <= 0 OR recency_days < 0 OR recency_days > 180
           ) FROM ml.customer_churn_dataset) AS invalid_feature_rows,
          (SELECT COUNT(*) FILTER (
              WHERE lifetime_delivered_orders < orders_180d
                 OR customer_tenure_days < recency_days
                 OR customer_tenure_days < 0
                 OR positive_interpurchase_gap_count < 0
                 OR zero_interpurchase_gap_share IS NULL
                 OR zero_interpurchase_gap_share < 0
                 OR zero_interpurchase_gap_share > 1
                 OR avg_interpurchase_days < 0
                 OR median_interpurchase_days < 0
                 OR interpurchase_gap_cv < 0
                 OR recency_to_avg_interpurchase_ratio < 0
                 OR (
                     positive_interpurchase_gap_count = 0
                     AND (
                         avg_interpurchase_days IS NOT NULL
                         OR median_interpurchase_days IS NOT NULL
                         OR interpurchase_gap_cv IS NOT NULL
                         OR recency_to_avg_interpurchase_ratio IS NOT NULL
                     )
                 )
                 OR (
                     positive_interpurchase_gap_count > 0
                     AND (
                         avg_interpurchase_days IS NULL
                         OR median_interpurchase_days IS NULL
                         OR interpurchase_gap_cv IS NULL
                         OR recency_to_avg_interpurchase_ratio IS NULL
                     )
                 )
                 OR orders_prior_180d < 0
                 OR revenue_prior_180d < 0
           ) FROM ml.customer_churn_dataset) AS invalid_history_feature_rows,
          (SELECT COUNT(*)
           FROM ml.customer_churn_dataset dataset
           CROSS JOIN source_end
           WHERE dataset.snapshot_date + INTERVAL '90 days'
                 > source_end.max_purchase_timestamp) AS incomplete_label_windows,
          (SELECT COUNT(*)
           FROM ml.customer_churn_dataset dataset
           CROSS JOIN LATERAL (
               SELECT AVG(order_review.order_avg_review_score)::numeric
                   AS expected_avg_review_score_180d
               FROM ml.delivered_orders d
               CROSS JOIN LATERAL (
                   SELECT AVG(r.review_score)::numeric AS order_avg_review_score
                   FROM ml.review_events r
                   WHERE r.order_id = d.order_id
                     AND r.review_answer_timestamp < dataset.snapshot_date
               ) order_review
               WHERE d.customer_unique_id = dataset.customer_unique_id
                 AND d.order_purchase_timestamp >= dataset.snapshot_date - INTERVAL '180 days'
                 AND d.order_purchase_timestamp < dataset.snapshot_date
                 AND d.order_delivered_customer_date < dataset.snapshot_date
                 AND order_review.order_avg_review_score IS NOT NULL
           ) expected
           WHERE dataset.avg_review_score_180d
                 IS DISTINCT FROM expected.expected_avg_review_score_180d
          ) AS review_feature_mismatches
    """)
    print("\nIntegrity checks (all counts should be zero)")
    print(checks.to_string(index=False))
    assert_zero_integrity_checks(checks.iloc[0].to_dict())
    print("\nAll churn dataset integrity checks passed.")


if __name__ == "__main__":
    main()
