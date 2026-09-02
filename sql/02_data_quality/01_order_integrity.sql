SELECT
    COUNT(*) AS orders,
    COUNT(DISTINCT order_id) AS unique_orders,
    COUNT(*) FILTER (WHERE order_id IS NULL) AS null_order_ids,
    COUNT(*) FILTER (WHERE customer_id IS NULL) AS null_customer_ids
FROM public.olist_orders;
