CREATE OR REPLACE VIEW analytics.customer_segments AS
SELECT
    segment,
    COUNT(*) AS customers,
    SUM(monetary) AS revenue,
    AVG(monetary) AS avg_customer_revenue,
    AVG(frequency) AS avg_orders
FROM analytics.customer_rfm
GROUP BY segment;
