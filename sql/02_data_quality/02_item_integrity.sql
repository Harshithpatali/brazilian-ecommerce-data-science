SELECT
    COUNT(*) AS item_rows,
    COUNT(DISTINCT (order_id, order_item_id)) AS unique_item_keys,
    COUNT(*) FILTER (WHERE product_id IS NULL) AS null_product_ids,
    COUNT(*) FILTER (WHERE seller_id IS NULL) AS null_seller_ids,
    COUNT(*) FILTER (WHERE price < 0) AS negative_prices,
    COUNT(*) FILTER (WHERE freight_value < 0) AS negative_freight
FROM public.olist_order_items;
