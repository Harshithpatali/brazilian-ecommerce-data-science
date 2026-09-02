CREATE OR REPLACE VIEW analytics.delivery_vs_review AS
SELECT
    CASE
        WHEN delivery_delay_days <= -1 THEN 'Early'
        WHEN delivery_delay_days <= 0 THEN 'On Time'
        WHEN delivery_delay_days <= 3 THEN '1-3 Days Late'
        WHEN delivery_delay_days <= 7 THEN '4-7 Days Late'
        ELSE '>7 Days Late'
    END AS delivery_bucket,
    COUNT(*) AS orders,
    AVG(avg_review_score) AS avg_review_score
FROM analytics.order_level
WHERE order_status = 'delivered'
  AND avg_review_score IS NOT NULL
  AND delivery_delay_days IS NOT NULL
GROUP BY 1
ORDER BY MIN(delivery_delay_days);
