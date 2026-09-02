SELECT
    COUNT(*) AS rows,
    COUNT(DISTINCT customer_unique_id) AS customers,
    COUNT(DISTINCT snapshot_date) AS snapshots,
    MIN(snapshot_date) AS first_snapshot,
    MAX(snapshot_date) AS last_snapshot,
    AVG(churned::numeric) AS churn_rate,
    MIN(orders_180d) AS min_orders_180d,
    MIN(recency_days) AS min_recency_days,
    MAX(recency_days) AS max_recency_days
FROM ml.customer_churn_dataset;

SELECT
    snapshot_date,
    COUNT(*) AS customers,
    AVG(churned::numeric) AS churn_rate
FROM ml.customer_churn_dataset
GROUP BY snapshot_date
ORDER BY snapshot_date;

SELECT
    churned,
    AVG(orders_180d) AS avg_orders_180d,
    AVG(orders_90d) AS avg_orders_90d,
    AVG(orders_30d) AS avg_orders_30d,
    AVG(revenue_180d) AS avg_revenue_180d,
    AVG(recency_days) AS avg_recency_days
FROM ml.customer_churn_dataset
group by churned
ORDER BY churned;
