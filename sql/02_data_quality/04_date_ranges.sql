SELECT
    MIN(order_purchase_timestamp) AS first_purchase,
    MAX(order_purchase_timestamp) AS last_purchase,
    MIN(order_approved_at) AS first_approval,
    MAX(order_approved_at) AS last_approval,
    COUNT(*) FILTER (
        WHERE order_delivered_customer_date < order_purchase_timestamp
    ) AS impossible_delivery_timestamps
FROM public.olist_orders;
