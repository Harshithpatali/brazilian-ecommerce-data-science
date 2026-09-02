CREATE OR REPLACE VIEW analytics.customer_metrics AS
SELECT
    customer_unique_id,
    COUNT(*) FILTER (WHERE order_status = 'delivered') AS orders,
    MIN(order_purchase_timestamp)::date AS first_order_date,
    MAX(order_purchase_timestamp)::date AS last_order_date,
    SUM(gross_order_value) FILTER (WHERE order_status = 'delivered') AS revenue,
    AVG(gross_order_value) FILTER (WHERE order_status = 'delivered') AS aov,
    AVG(avg_review_score) AS avg_review_score,
    AVG(delivery_delay_days) AS avg_delivery_delay_days
FROM analytics.order_level
GROUP BY customer_unique_id;
