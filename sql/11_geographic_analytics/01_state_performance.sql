CREATE OR REPLACE VIEW analytics.geographic_performance AS
SELECT
    c.customer_state,
    COUNT(DISTINCT o.order_id) AS orders,
    COUNT(DISTINCT c.customer_unique_id) AS customers,
    SUM(ol.gross_order_value) AS revenue,
    AVG(ol.gross_order_value) AS aov,
    AVG(ol.delivery_days) AS avg_delivery_days,
    AVG(ol.avg_review_score) AS avg_review_score
FROM analytics.order_level ol
JOIN public.olist_customers c
  ON c.customer_id = ol.customer_id
JOIN public.olist_orders o
  ON o.order_id = ol.order_id
WHERE ol.order_status = 'delivered'
GROUP BY c.customer_state
ORDER BY revenue DESC;
