/*
ENGAGED CUSTOMER CHURN DATASET

Purpose
-------
Additional ML layer for the SQL-first Olist project.

Grain
-----
One row per customer-month prediction snapshot.

Population
----------
Customers who have reached their second delivered order.

Snapshot logic
--------------
Create monthly snapshots from the customer's second-order month through
its first observed 90-day churn event, or through the global observation
cutoff when no churn event occurs. Months with no purchase are retained.

Target
------
churned = 1 when the customer has no delivered purchase in the next 90 days.

Leakage control
---------------
Features use only information observed on or before the activity month.
Only snapshots with a complete 90-day future observation window are kept.
*/

/* =========================================================
   1. CLEANUP
   These objects are always materialized as TABLEs (CREATE TABLE ... AS)
   in this script, never as views. On a re-run they already exist as
   tables, so a "DROP VIEW IF EXISTS" on them raises WrongObjectType
   (IF EXISTS only swallows "does not exist", not "wrong type").
   DROP TABLE IF EXISTS is therefore sufficient and safe on a fresh DB too.
   ========================================================= */
DROP TABLE IF EXISTS ml.delivered_orders CASCADE;
DROP TABLE IF EXISTS ml.customer_monthly_activity CASCADE;
DROP TABLE IF EXISTS ml.customer_snapshot_calendar CASCADE;
DROP TABLE IF EXISTS ml.customer_churn_dataset CASCADE;

/* =========================================================
   2. DELIVERED ORDERS
   ========================================================= */
CREATE TABLE ml.delivered_orders AS
SELECT
    customer_unique_id,
    order_id,
    order_purchase_timestamp,
    gross_order_value,
    avg_review_score,
    delivery_delay_days
FROM analytics.order_level
WHERE order_status = 'delivered'
  AND customer_unique_id IS NOT NULL;

CREATE INDEX idx_ml_delivered_customer_time
    ON ml.delivered_orders (customer_unique_id, order_purchase_timestamp);
CREATE INDEX idx_ml_delivered_time
    ON ml.delivered_orders (order_purchase_timestamp);
ANALYZE ml.delivered_orders;

/* =========================================================
   3. CUSTOMER-MONTH ACTIVITY
   One row per customer per month with a delivered purchase.
   ========================================================= */
CREATE TABLE ml.customer_monthly_activity AS
SELECT
    customer_unique_id,
    DATE_TRUNC('month', order_purchase_timestamp)::date AS activity_month,
    COUNT(DISTINCT order_id) AS orders,
    SUM(COALESCE(gross_order_value, 0)) AS revenue,
    AVG(gross_order_value) AS avg_order_value,
    AVG(avg_review_score) AS avg_review_score,
    AVG(delivery_delay_days) AS avg_delivery_delay,
    MIN(order_purchase_timestamp) AS first_order_timestamp,
    MAX(order_purchase_timestamp) AS last_order_timestamp
FROM ml.delivered_orders
GROUP BY customer_unique_id,
         DATE_TRUNC('month', order_purchase_timestamp)::date;

CREATE INDEX idx_ml_customer_month
    ON ml.customer_monthly_activity (customer_unique_id, activity_month);
ANALYZE ml.customer_monthly_activity;

/* =========================================================
   4. GLOBAL OBSERVATION CUTOFF
   A snapshot must have a full 90-day future window.
   ========================================================= */
CREATE TABLE ml.customer_snapshot_calendar AS
WITH ranked_orders AS (
    SELECT
        customer_unique_id,
        order_id,
        order_purchase_timestamp,
        ROW_NUMBER() OVER (
            PARTITION BY customer_unique_id
            ORDER BY order_purchase_timestamp, order_id
        ) AS order_number
    FROM ml.delivered_orders
),
second_order_month AS (
    SELECT
        customer_unique_id,
        DATE_TRUNC('month', order_purchase_timestamp)::date AS second_order_month
    FROM ranked_orders
    WHERE order_number = 2
),
cutoff AS (
    SELECT
        (MAX(order_purchase_timestamp)::date - INTERVAL '90 days')::date AS max_snapshot_date,
        DATE_TRUNC(
            'month',
            MAX(order_purchase_timestamp)::date - INTERVAL '90 days'
        )::date AS max_snapshot_month
    FROM ml.delivered_orders
)
SELECT
    s.customer_unique_id,
    calendar.snapshot_month,
    COALESCE(a.orders, 0) AS orders,
    COALESCE(a.revenue, 0) AS revenue,
    a.avg_order_value,
    a.avg_review_score,
    a.avg_delivery_delay,
    a.first_order_timestamp,
    a.last_order_timestamp
FROM second_order_month s
CROSS JOIN cutoff c
CROSS JOIN LATERAL generate_series(
    s.second_order_month,
    c.max_snapshot_month,
    INTERVAL '1 month'
) AS calendar(snapshot_month)
LEFT JOIN ml.customer_monthly_activity a
    ON a.customer_unique_id = s.customer_unique_id
   AND a.activity_month = calendar.snapshot_month
WHERE calendar.snapshot_month < c.max_snapshot_month
   OR (
       calendar.snapshot_month = c.max_snapshot_month
       AND c.max_snapshot_date >= (calendar.snapshot_month + INTERVAL '1 month')::date
   );

CREATE INDEX idx_ml_snapshot_calendar
    ON ml.customer_snapshot_calendar (customer_unique_id, snapshot_month);
ANALYZE ml.customer_snapshot_calendar;

/* =========================================================
   5. BUILD POINT-IN-TIME FEATURES AND LABELS
   ========================================================= */
CREATE TABLE ml.customer_churn_dataset AS
WITH enriched AS (
    SELECT
        c.*,

        /* Most recent real purchase observed by the snapshot month. */
        MAX(c.last_order_timestamp) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY c.snapshot_month
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        ) AS carried_last_order_timestamp,

        /* First actual purchase in a future calendar month. */
        MIN(c.first_order_timestamp) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY c.snapshot_month
            ROWS BETWEEN 1 FOLLOWING AND UNBOUNDED FOLLOWING
        ) AS next_order_timestamp,

        /* Rolling 180-day behavior. */
        SUM(c.orders) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY c.snapshot_month
            RANGE BETWEEN INTERVAL '179 days' PRECEDING AND CURRENT ROW
        ) AS orders_180d,
        SUM(c.revenue) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY c.snapshot_month
            RANGE BETWEEN INTERVAL '179 days' PRECEDING AND CURRENT ROW
        ) AS revenue_180d,
        AVG(c.avg_order_value) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY c.snapshot_month
            RANGE BETWEEN INTERVAL '179 days' PRECEDING AND CURRENT ROW
        ) AS avg_order_value_180d,
        AVG(c.avg_review_score) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY c.snapshot_month
            RANGE BETWEEN INTERVAL '179 days' PRECEDING AND CURRENT ROW
        ) AS avg_review_score_180d,
        AVG(c.avg_delivery_delay) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY c.snapshot_month
            RANGE BETWEEN INTERVAL '179 days' PRECEDING AND CURRENT ROW
        ) AS avg_delivery_delay_180d,
        COUNT(*) FILTER (WHERE c.orders > 0) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY c.snapshot_month
            RANGE BETWEEN INTERVAL '179 days' PRECEDING AND CURRENT ROW
        ) AS active_months_180d,

        /* Rolling 90-day behavior. */
        SUM(c.orders) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY c.snapshot_month
            RANGE BETWEEN INTERVAL '89 days' PRECEDING AND CURRENT ROW
        ) AS orders_90d,
        SUM(c.revenue) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY c.snapshot_month
            RANGE BETWEEN INTERVAL '89 days' PRECEDING AND CURRENT ROW
        ) AS revenue_90d,

        /* Rolling 30-day behavior. */
        SUM(c.orders) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY c.snapshot_month
            RANGE BETWEEN INTERVAL '29 days' PRECEDING AND CURRENT ROW
        ) AS orders_30d,
        SUM(c.revenue) OVER (
            PARTITION BY c.customer_unique_id
            ORDER BY c.snapshot_month
            RANGE BETWEEN INTERVAL '29 days' PRECEDING AND CURRENT ROW
        ) AS revenue_30d
    FROM ml.customer_snapshot_calendar c
),
base AS (
    SELECT
        customer_unique_id,
        (snapshot_month + INTERVAL '1 month')::date AS snapshot_date,
        COALESCE(orders_180d, 0) AS orders_180d,
        COALESCE(orders_90d, 0) AS orders_90d,
        COALESCE(orders_30d, 0) AS orders_30d,
        COALESCE(revenue_180d, 0) AS revenue_180d,
        COALESCE(revenue_90d, 0) AS revenue_90d,
        COALESCE(revenue_30d, 0) AS revenue_30d,
        avg_order_value_180d,
        avg_review_score_180d,
        avg_delivery_delay_180d,
        COALESCE(active_months_180d, 0) AS active_months_180d,
        carried_last_order_timestamp::date AS last_order_date,
        (
            (snapshot_month + INTERVAL '1 month')::date
            - carried_last_order_timestamp::date
        )::int AS recency_days,
        CASE
            WHEN COALESCE(orders_180d, 0) > 0
            THEN COALESCE(orders_90d, 0)::numeric / NULLIF(orders_180d, 0)
            ELSE 0
        END AS recent_order_share,
        CASE
            WHEN COALESCE(revenue_180d, 0) > 0
            THEN COALESCE(revenue_90d, 0) / NULLIF(revenue_180d, 0)
            ELSE 0
        END AS recent_revenue_share,
        CASE
            WHEN COALESCE(active_months_180d, 0) > 0
            THEN COALESCE(orders_180d, 0)::numeric / NULLIF(active_months_180d, 0)
            ELSE 0
        END AS orders_per_active_month,
        CASE
            WHEN COALESCE(orders_180d, 0) > 0
            THEN COALESCE(revenue_180d, 0) / NULLIF(orders_180d, 0)
            ELSE 0
        END AS revenue_per_order,
        CASE
            WHEN COALESCE(orders_180d, 0) > COALESCE(orders_90d, 0)
            THEN COALESCE(orders_90d, 0)::numeric /
                 NULLIF(orders_180d - orders_90d, 0)
            ELSE NULL
        END AS recent_vs_older_order_ratio,
        next_order_timestamp
    FROM enriched
),
labeled AS (
    SELECT
        b.*,
        CASE
            WHEN b.next_order_timestamp IS NULL THEN 1
            WHEN b.next_order_timestamp >= b.snapshot_date + INTERVAL '90 days' THEN 1
            ELSE 0
        END AS churned
    FROM base b
),
first_churn AS (
    SELECT
        l.*,
        MIN(snapshot_date) FILTER (WHERE churned = 1) OVER (
            PARTITION BY customer_unique_id
        ) AS first_churn_snapshot
    FROM labeled l
)
SELECT
    customer_unique_id,
    snapshot_date,
    orders_180d,
    orders_90d,
    orders_30d,
    revenue_180d,
    revenue_90d,
    revenue_30d,
    avg_order_value_180d,
    avg_review_score_180d,
    avg_delivery_delay_180d,
    active_months_180d,
    last_order_date,
    recency_days,
    recent_order_share,
    recent_revenue_share,
    orders_per_active_month,
    revenue_per_order,
    recent_vs_older_order_ratio,
    churned
FROM first_churn
WHERE first_churn_snapshot IS NULL
   OR snapshot_date <= first_churn_snapshot;

/* =========================================================
   6. INDEXES AND STATISTICS
   ========================================================= */
CREATE INDEX idx_ml_churn_snapshot
    ON ml.customer_churn_dataset (snapshot_date);
CREATE INDEX idx_ml_churn_customer_snapshot
    ON ml.customer_churn_dataset (customer_unique_id, snapshot_date);
CREATE INDEX idx_ml_churn_target
    ON ml.customer_churn_dataset (churned);
ANALYZE ml.customer_churn_dataset;
