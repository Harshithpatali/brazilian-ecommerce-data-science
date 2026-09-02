CREATE OR REPLACE VIEW analytics.delivery_performance AS
SELECT
    DATE_TRUNC('month', order_purchase_timestamp)::date AS month,
    COUNT(*) FILTER (
        WHERE order_status = 'delivered'
          AND order_delivered_customer_date IS NOT NULL
    ) AS delivered_orders,
    AVG(delivery_days) FILTER (
        WHERE order_status = 'delivered'
          AND delivery_days IS NOT NULL
    ) AS avg_delivery_days,
    AVG(delivery_delay_days) FILTER (
        WHERE order_status = 'delivered'
          AND delivery_delay_days IS NOT NULL
    ) AS avg_delivery_delay_days,
    AVG(
        CASE
            WHEN delivery_delay_days > 0 THEN 1.0 ELSE 0.0
        END
    ) FILTER (WHERE order_status = 'delivered') AS late_order_rate
FROM analytics.order_level
GROUP BY 1
ORDER BY 1;
