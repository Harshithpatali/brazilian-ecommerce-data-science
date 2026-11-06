/* Point-in-time churn dataset integrity checks. */

SELECT
    COUNT(*) AS rows,
    COUNT(DISTINCT customer_unique_id) AS customers,
    COUNT(DISTINCT snapshot_date) AS snapshots,
    MIN(snapshot_date) AS first_snapshot,
    MAX(snapshot_date) AS last_snapshot,
    AVG(churned::numeric) AS churn_rate,
    SUM(CASE WHEN churned = 1 THEN 1 ELSE 0 END) AS churned_rows,
    SUM(CASE WHEN churned = 0 THEN 1 ELSE 0 END) AS non_churned_rows
FROM ml.customer_churn_dataset;

SELECT
    snapshot_date,
    COUNT(*) AS customers,
    SUM(CASE WHEN churned = 1 THEN 1 ELSE 0 END) AS churned,
    SUM(CASE WHEN churned = 0 THEN 1 ELSE 0 END) AS non_churned,
    AVG(churned::numeric) AS churn_rate
FROM ml.customer_churn_dataset
GROUP BY snapshot_date
ORDER BY snapshot_date;

/* Expected: no rows. */
SELECT customer_unique_id, snapshot_date, COUNT(*) AS duplicate_count
FROM ml.customer_churn_dataset
GROUP BY customer_unique_id, snapshot_date
HAVING COUNT(*) > 1;

/* Expected target_mismatch_count: zero. */
SELECT COUNT(*) FILTER (
    WHERE dataset.churned <> CASE
        WHEN EXISTS (
            SELECT 1
            FROM ml.delivered_orders future_order
            WHERE future_order.customer_unique_id = dataset.customer_unique_id
              AND future_order.order_purchase_timestamp >= dataset.snapshot_date
              AND future_order.order_purchase_timestamp
                    < dataset.snapshot_date + INTERVAL '90 days'
        ) THEN 0 ELSE 1
    END
) AS target_mismatch_count
FROM ml.customer_churn_dataset dataset;

/* Expected invalid_feature_rows: zero. */
SELECT COUNT(*) FILTER (
    WHERE dataset.orders_180d <= 0
       OR dataset.recency_days < 0
       OR dataset.recency_days > 180
) AS invalid_feature_rows
FROM ml.customer_churn_dataset dataset;

/* Expected incomplete_window_rows: zero. */
WITH source_end AS (
    SELECT MAX(order_purchase_timestamp) AS max_purchase_timestamp
    FROM analytics.order_level
)
SELECT COUNT(*) AS incomplete_window_rows
FROM ml.customer_churn_dataset dataset
CROSS JOIN source_end
WHERE dataset.snapshot_date + INTERVAL '90 days'
      > source_end.max_purchase_timestamp;

/* Directional diagnostic by target. */
SELECT
    churned,
    COUNT(*) AS rows,
    AVG(orders_180d) AS avg_orders_180d,
    AVG(orders_90d) AS avg_orders_90d,
    AVG(orders_30d) AS avg_orders_30d,
    AVG(revenue_180d) AS avg_revenue_180d,
    AVG(recency_days) AS avg_recency_days
FROM ml.customer_churn_dataset
GROUP BY churned
ORDER BY churned;


/* Expected review_feature_mismatch_count: zero.
   Recompute an order-weighted average using only reviews answered by the
   snapshot. This detects use of review scores before they were observable. */
SELECT COUNT(*) AS review_feature_mismatch_count
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
      IS DISTINCT FROM expected.expected_avg_review_score_180d;
