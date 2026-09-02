SELECT 'olist_customers' AS table_name, COUNT(*) AS row_count FROM public.olist_customers
UNION ALL SELECT 'olist_orders', COUNT(*) FROM public.olist_orders
UNION ALL SELECT 'olist_order_items', COUNT(*) FROM public.olist_order_items
UNION ALL SELECT 'olist_order_payments', COUNT(*) FROM public.olist_order_payments
UNION ALL SELECT 'olist_order_reviews', COUNT(*) FROM public.olist_order_reviews
UNION ALL SELECT 'olist_products', COUNT(*) FROM public.olist_products
UNION ALL SELECT 'olist_sellers', COUNT(*) FROM public.olist_sellers
UNION ALL SELECT 'olist_geolocation', COUNT(*) FROM public.olist_geolocation
UNION ALL SELECT 'product_category_name_translation', COUNT(*) FROM public.product_category_name_translation
ORDER BY row_count DESC;
