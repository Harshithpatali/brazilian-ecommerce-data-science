CREATE OR REPLACE VIEW analytics.monthly_revenue AS
SELECT
    DATE_TRUNC('month', order_purchase_timestamp)::date AS month,
    COUNT(*) FILTER (WHERE order_status = 'delivered') AS orders,
    COUNT(DISTINCT customer_unique_id) FILTER (WHERE order_status = 'delivered') AS customers,
    SUM(product_revenue) FILTER (WHERE order_status = 'delivered') AS revenue,
    SUM(freight_revenue) FILTER (WHERE order_status = 'delivered') AS freight_revenue,
    AVG(gross_order_value) FILTER (WHERE order_status = 'delivered') AS aov
FROM analytics.order_level
GROUP BY 1
ORDER BY 1;
