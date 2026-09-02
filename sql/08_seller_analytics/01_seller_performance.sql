CREATE OR REPLACE VIEW analytics.seller_performance AS
WITH review AS (
    SELECT order_id, AVG(review_score) AS avg_review_score
    FROM public.olist_order_reviews
    GROUP BY order_id
)
SELECT
    oi.seller_id,
    COUNT(DISTINCT oi.order_id) AS orders,
    COUNT(*) AS items,
    SUM(oi.price) AS revenue,
    AVG(oi.price) AS avg_item_price,
    AVG(review.avg_review_score) AS avg_review_score,
    AVG(
        CASE WHEN o.order_delivered_customer_date IS NOT NULL
        THEN EXTRACT(EPOCH FROM (
            o.order_delivered_customer_date - o.order_estimated_delivery_date
        )) / 86400.0 END
    ) AS avg_delivery_delay_days
FROM public.olist_order_items oi
JOIN public.olist_orders o USING (order_id)
LEFT JOIN review USING (order_id)
WHERE o.order_status = 'delivered'
GROUP BY oi.seller_id
ORDER BY revenue DESC;
