SELECT 'items_without_orders' AS check_name, COUNT(*) AS bad_rows
FROM public.olist_order_items i
LEFT JOIN public.olist_orders o USING (order_id)
WHERE o.order_id IS NULL
UNION ALL
SELECT 'items_without_products', COUNT(*)
FROM public.olist_order_items i
LEFT JOIN public.olist_products p USING (product_id)
WHERE p.product_id IS NULL
UNION ALL
SELECT 'items_without_sellers', COUNT(*)
FROM public.olist_order_items i
LEFT JOIN public.olist_sellers s USING (seller_id)
WHERE s.seller_id IS NULL;
