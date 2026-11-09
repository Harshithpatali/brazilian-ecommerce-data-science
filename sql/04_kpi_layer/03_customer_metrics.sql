CREATE OR REPLACE VIEW analytics.customer_metrics AS
WITH delivered_orders AS (
    SELECT
        customer_unique_id,
        order_purchase_timestamp::date AS purchase_date,
        ROW_NUMBER() OVER (
            PARTITION BY customer_unique_id
            ORDER BY order_purchase_timestamp, order_id
        ) AS purchase_number
    FROM analytics.order_level
    WHERE order_status = 'delivered'
),
second_purchase AS (
    SELECT customer_unique_id, purchase_date AS second_order_date
    FROM delivered_orders
    WHERE purchase_number = 2
)
SELECT
    m.customer_unique_id,
    COUNT(*) FILTER (WHERE m.order_status = 'delivered') AS orders,
    MIN(m.order_purchase_timestamp)::date AS first_order_date,
    MAX(m.order_purchase_timestamp)::date AS last_order_date,
    s.second_order_date,
    SUM(m.gross_order_value) FILTER (WHERE m.order_status = 'delivered') AS revenue,
    AVG(m.gross_order_value) FILTER (WHERE m.order_status = 'delivered') AS aov,
    AVG(m.avg_review_score) AS avg_review_score,
    AVG(m.delivery_delay_days) AS avg_delivery_delay_days
FROM analytics.order_level m
LEFT JOIN second_purchase s USING (customer_unique_id)
GROUP BY m.customer_unique_id, s.second_order_date;
