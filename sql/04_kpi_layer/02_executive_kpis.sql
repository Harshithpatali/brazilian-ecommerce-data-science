CREATE OR REPLACE VIEW analytics.executive_kpis AS
SELECT
    SUM(product_revenue) FILTER (WHERE order_status = 'delivered') AS revenue,
    COUNT(*) FILTER (WHERE order_status = 'delivered') AS orders,
    COUNT(DISTINCT customer_unique_id) FILTER (WHERE order_status = 'delivered') AS customers,
    AVG(gross_order_value) FILTER (WHERE order_status = 'delivered') AS aov
FROM analytics.order_level;
