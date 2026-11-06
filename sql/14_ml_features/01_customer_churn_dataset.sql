/*
POINT-IN-TIME CUSTOMER CHURN DATASET

Target interval: [snapshot_date, snapshot_date + 90 days)
Features use only orders purchased before the snapshot and delivered to the
customer before the snapshot. Review scores are available only after
review_answer_timestamp. The source observation end uses all source orders.
Every eligible monthly snapshot is retained: 90-day churn is not permanent.
*/

DROP TABLE IF EXISTS ml.delivered_orders CASCADE;
DROP TABLE IF EXISTS ml.customer_monthly_activity CASCADE;
DROP TABLE IF EXISTS ml.customer_snapshot_calendar CASCADE;
DROP TABLE IF EXISTS ml.customer_churn_dataset CASCADE;
DROP TABLE IF EXISTS ml.review_events CASCADE;

CREATE TABLE ml.delivered_orders AS
SELECT
    customer_unique_id,
    order_id,
    order_purchase_timestamp,
    order_delivered_customer_date,
    COALESCE(gross_order_value, 0)::numeric AS gross_order_value,
    delivery_delay_days
FROM analytics.order_level
WHERE order_status = 'delivered'
  AND customer_unique_id IS NOT NULL
  AND order_purchase_timestamp IS NOT NULL
  AND order_delivered_customer_date IS NOT NULL;

CREATE INDEX idx_ml_delivered_customer_purchase
    ON ml.delivered_orders (customer_unique_id, order_purchase_timestamp);
CREATE INDEX idx_ml_delivered_customer_delivery
    ON ml.delivered_orders (customer_unique_id, order_delivered_customer_date);
CREATE INDEX idx_ml_delivered_purchase
    ON ml.delivered_orders (order_purchase_timestamp);
ANALYZE ml.delivered_orders;

/* Point-in-time review events support efficient as-of review features. */
CREATE TABLE ml.review_events AS
SELECT order_id, review_score, review_creation_date, review_answer_timestamp
FROM public.olist_order_reviews
WHERE order_id IS NOT NULL AND review_answer_timestamp IS NOT NULL;
CREATE INDEX idx_ml_review_order_date
    ON ml.review_events (order_id, review_answer_timestamp);
ANALYZE ml.review_events;

CREATE TABLE ml.customer_monthly_activity AS
SELECT
    customer_unique_id,
    DATE_TRUNC('month', order_purchase_timestamp)::date AS activity_month,
    COUNT(*) AS orders,
    SUM(gross_order_value) AS revenue,
    AVG(gross_order_value) AS avg_order_value,
    AVG(delivery_delay_days) AS avg_delivery_delay,
    MIN(order_purchase_timestamp) AS first_order_timestamp,
    MAX(order_purchase_timestamp) AS last_order_timestamp
FROM ml.delivered_orders
GROUP BY customer_unique_id, DATE_TRUNC('month', order_purchase_timestamp)::date;

CREATE INDEX idx_ml_customer_month
    ON ml.customer_monthly_activity (customer_unique_id, activity_month);
ANALYZE ml.customer_monthly_activity;

/* Begin after the second delivered order. Only dates with a fully observed
   90-day future label window are materialized. */
CREATE TABLE ml.customer_snapshot_calendar AS
WITH ranked_delivered AS (
    SELECT
        customer_unique_id,
        order_id,
        order_delivered_customer_date,
        ROW_NUMBER() OVER (
            PARTITION BY customer_unique_id
            ORDER BY order_delivered_customer_date, order_id
        ) AS delivered_order_number
    FROM ml.delivered_orders
),
second_delivered AS (
    SELECT
        customer_unique_id,
        order_delivered_customer_date AS second_delivery_timestamp
    FROM ranked_delivered
    WHERE delivered_order_number = 2
),
observation_end AS (
    SELECT MAX(order_purchase_timestamp) AS max_observed_purchase_timestamp
    FROM analytics.order_level
    WHERE order_purchase_timestamp IS NOT NULL
)
SELECT
    s.customer_unique_id,
    generated.snapshot_timestamp::date AS snapshot_date
FROM second_delivered s
CROSS JOIN observation_end o
CROSS JOIN LATERAL GENERATE_SERIES(
    (
        DATE_TRUNC('month', s.second_delivery_timestamp)::date
        + INTERVAL '1 month'
    )::timestamp,
    DATE_TRUNC('month', o.max_observed_purchase_timestamp)::timestamp,
    INTERVAL '1 month'
) AS generated(snapshot_timestamp)
WHERE o.max_observed_purchase_timestamp IS NOT NULL
  AND generated.snapshot_timestamp + INTERVAL '90 days'
      <= o.max_observed_purchase_timestamp;

CREATE UNIQUE INDEX idx_ml_snapshot_customer_date
    ON ml.customer_snapshot_calendar (customer_unique_id, snapshot_date);
CREATE INDEX idx_ml_snapshot_date
    ON ml.customer_snapshot_calendar (snapshot_date);
ANALYZE ml.customer_snapshot_calendar;

/* Point-in-time features and order-timestamp-based forward target. */
CREATE TABLE ml.customer_churn_dataset AS
SELECT
    s.customer_unique_id,
    s.snapshot_date,
    f.orders_180d,
    f.orders_90d,
    f.orders_30d,
    f.revenue_180d,
    f.revenue_90d,
    f.revenue_30d,
    f.avg_order_value_180d,
    review.avg_review_score_180d,
    f.avg_delivery_delay_180d,
    f.active_months_180d,
    f.last_order_date,
    f.recency_days,
    f.orders_90d::numeric / NULLIF(f.orders_180d, 0) AS recent_order_share,
    f.revenue_90d / NULLIF(f.revenue_180d, 0) AS recent_revenue_share,
    f.orders_180d::numeric / NULLIF(f.active_months_180d, 0)
        AS orders_per_active_month,
    f.revenue_180d / NULLIF(f.orders_180d, 0) AS revenue_per_order,
    CASE
        WHEN f.orders_180d > f.orders_90d
        THEN f.orders_90d::numeric
             / NULLIF(f.orders_180d - f.orders_90d, 0)
        ELSE NULL
    END AS recent_vs_older_order_ratio,
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM ml.delivered_orders future_order
            WHERE future_order.customer_unique_id = s.customer_unique_id
              AND future_order.order_purchase_timestamp >= s.snapshot_date
              AND future_order.order_purchase_timestamp
                    < s.snapshot_date + INTERVAL '90 days'
        ) THEN 0
        ELSE 1
    END AS churned
FROM ml.customer_snapshot_calendar s
CROSS JOIN LATERAL (
    SELECT
        COUNT(*)::integer AS orders_180d,
        COUNT(*) FILTER (
            WHERE d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '90 days'
        )::integer AS orders_90d,
        COUNT(*) FILTER (
            WHERE d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '30 days'
        )::integer AS orders_30d,
        COALESCE(SUM(d.gross_order_value), 0)::numeric AS revenue_180d,
        COALESCE(
            SUM(d.gross_order_value) FILTER (
                WHERE d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '90 days'
            ), 0
        )::numeric AS revenue_90d,
        COALESCE(
            SUM(d.gross_order_value) FILTER (
                WHERE d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '30 days'
            ), 0
        )::numeric AS revenue_30d,
        AVG(d.gross_order_value)::numeric AS avg_order_value_180d,
        AVG(d.delivery_delay_days)::numeric AS avg_delivery_delay_180d,
        COUNT(DISTINCT DATE_TRUNC('month', d.order_purchase_timestamp))::integer
            AS active_months_180d,
        MAX(d.order_purchase_timestamp)::date AS last_order_date,
        (s.snapshot_date - MAX(d.order_purchase_timestamp)::date)::integer
            AS recency_days
    FROM ml.delivered_orders d
    WHERE d.customer_unique_id = s.customer_unique_id
      AND d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '180 days'
      AND d.order_purchase_timestamp < s.snapshot_date
      AND d.order_delivered_customer_date < s.snapshot_date
) f
LEFT JOIN LATERAL (
    /* Average reviews per order first, then weight each order equally.
       Multiple review rows for one order must not dominate this feature. */
    SELECT AVG(order_review.order_avg_review_score)::numeric
        AS avg_review_score_180d
    FROM ml.delivered_orders d
    CROSS JOIN LATERAL (
        SELECT AVG(r.review_score)::numeric AS order_avg_review_score
        FROM ml.review_events r
        WHERE r.order_id = d.order_id
          AND r.review_answer_timestamp < s.snapshot_date
    ) order_review
    WHERE d.customer_unique_id = s.customer_unique_id
      AND d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '180 days'
      AND d.order_purchase_timestamp < s.snapshot_date
      AND d.order_delivered_customer_date < s.snapshot_date
      AND order_review.order_avg_review_score IS NOT NULL
) review ON TRUE
WHERE f.orders_180d > 0;

CREATE UNIQUE INDEX idx_ml_churn_customer_snapshot
    ON ml.customer_churn_dataset (customer_unique_id, snapshot_date);
CREATE INDEX idx_ml_churn_snapshot
    ON ml.customer_churn_dataset (snapshot_date);
CREATE INDEX idx_ml_churn_target
    ON ml.customer_churn_dataset (churned);
ANALYZE ml.customer_churn_dataset;
