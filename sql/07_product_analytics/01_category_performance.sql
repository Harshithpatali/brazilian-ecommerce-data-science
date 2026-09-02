CREATE OR REPLACE VIEW analytics.category_performance AS
SELECT
    COALESCE(t.product_category_name_english, p.product_category_name, 'Unknown') AS category,
    COUNT(DISTINCT oi.order_id) AS orders,
    COUNT(*) AS items,
    SUM(oi.price) AS revenue,
    AVG(oi.price) AS avg_item_price,
    AVG(r.review_score) AS avg_review_score
FROM public.olist_order_items oi
JOIN public.olist_orders o USING (order_id)
LEFT JOIN public.olist_products p USING (product_id)
LEFT JOIN public.product_category_name_translation t
  ON t.product_category_name = p.product_category_name
LEFT JOIN public.olist_order_reviews r USING (order_id)
WHERE o.order_status = 'delivered'
GROUP BY 1
ORDER BY revenue DESC;
