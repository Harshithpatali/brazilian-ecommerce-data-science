SELECT
    COUNT(*) AS rows,
    COUNT(DISTINCT customer_unique_id) AS customers,
    COUNT(DISTINCT snapshot_date) AS snapshots,
    MIN(snapshot_date) AS first_snapshot,
    MAX(snapshot_date) AS last_snapshot,
    ROUND(AVG(churned::numeric), 4) AS churn_rate,
    MIN(orders_180d) AS min_orders_180d,
    MAX(orders_180d) AS max_orders_180d,
    MIN(recency_days) AS min_recency_days
FROM ml.customer_churn_dataset;

SELECT
    churned,
    COUNT(*) AS rows,
    ROUND(AVG(orders_180d), 3) AS avg_orders_180d,
    ROUND(AVG(orders_90d), 3) AS avg_orders_90d,
    ROUND(AVG(revenue_180d), 2) AS avg_revenue_180d,
    ROUND(AVG(recency_days), 2) AS avg_recency_days
FROM ml.customer_churn_dataset
GROUP BY churned
ORDER BY churned;
