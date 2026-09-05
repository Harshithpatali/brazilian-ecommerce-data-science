CREATE OR REPLACE VIEW analytics.customer_anova_dataset AS
WITH customer_experience AS (
    SELECT
        customer_unique_id,
        AVG(avg_review_score) AS avg_review_score,
        AVG(delivery_delay_days) AS avg_delivery_delay_days,
        COUNT(*) AS delivered_orders
    FROM analytics.order_level
    WHERE order_status = 'delivered'
    GROUP BY customer_unique_id
)
SELECT
    r.customer_unique_id,
    r.segment,
    r.frequency,
    r.monetary,
    e.avg_review_score,
    e.avg_delivery_delay_days,
    e.delivered_orders,
    CASE
        WHEN e.avg_delivery_delay_days <= 0 THEN 'On Time / Early'
        WHEN e.avg_delivery_delay_days <= 3 THEN '1-3 Days Late'
        WHEN e.avg_delivery_delay_days <= 7 THEN '4-7 Days Late'
        ELSE '8+ Days Late'
    END AS delivery_performance_group
FROM analytics.customer_rfm r
JOIN customer_experience e
  ON e.customer_unique_id = r.customer_unique_id
WHERE r.monetary IS NOT NULL
  AND r.frequency IS NOT NULL
  AND e.avg_review_score IS NOT NULL
  AND e.avg_delivery_delay_days IS NOT NULL;
