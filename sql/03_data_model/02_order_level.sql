CREATE OR REPLACE VIEW analytics.order_level AS
WITH item_agg AS (
    SELECT
        order_id,
        COUNT(*) AS item_count,
        SUM(price) AS product_revenue,
        SUM(freight_value) AS freight_revenue,
        SUM(price + freight_value) AS gross_order_value
    FROM public.olist_order_items
    GROUP BY order_id
),
payment_agg AS (
    SELECT
        order_id,
        SUM(payment_value) AS payment_value,
        COUNT(*) AS payment_count
    FROM public.olist_order_payments
    GROUP BY order_id
),
review_agg AS (
    SELECT
        order_id,
        AVG(review_score) AS avg_review_score,
        COUNT(*) AS review_count
    FROM public.olist_order_reviews
    GROUP BY order_id
)
SELECT
    o.order_id,
    c.customer_unique_id,
    o.customer_id,
    o.order_status,
    o.order_purchase_timestamp,
    o.order_approved_at,
    o.order_delivered_carrier_date,
    o.order_delivered_customer_date,
    o.order_estimated_delivery_date,
    COALESCE(i.item_count, 0) AS item_count,
    COALESCE(i.product_revenue, 0) AS product_revenue,
    COALESCE(i.freight_revenue, 0) AS freight_revenue,
    COALESCE(i.gross_order_value, 0) AS gross_order_value,
    COALESCE(p.payment_value, 0) AS payment_value,
    COALESCE(p.payment_count, 0) AS payment_count,
    r.avg_review_score,
    COALESCE(r.review_count, 0) AS review_count,
    CASE
        WHEN o.order_delivered_customer_date IS NOT NULL
        THEN EXTRACT(EPOCH FROM (
            o.order_delivered_customer_date - o.order_purchase_timestamp
        )) / 86400.0
    END AS delivery_days,
    CASE
        WHEN o.order_delivered_customer_date IS NOT NULL
             AND o.order_estimated_delivery_date IS NOT NULL
        THEN EXTRACT(EPOCH FROM (
            o.order_delivered_customer_date - o.order_estimated_delivery_date
        )) / 86400.0
    END AS delivery_delay_days
FROM public.olist_orders o
JOIN public.olist_customers c
  ON c.customer_id = o.customer_id
LEFT JOIN item_agg i USING (order_id)
LEFT JOIN payment_agg p USING (order_id)
LEFT JOIN review_agg r USING (order_id);
