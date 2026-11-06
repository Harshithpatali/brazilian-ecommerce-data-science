from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.io import query_to_dataframe


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


if __name__ == "__main__":
    main()
