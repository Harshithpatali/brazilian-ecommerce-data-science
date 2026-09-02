CREATE OR REPLACE VIEW analytics.delivery_review_dataset AS
SELECT
    delivery_delay_days,
    avg_review_score,
    delivery_days,
    gross_order_value
FROM analytics.order_level
WHERE order_status = 'delivered'
  AND delivery_delay_days IS NOT NULL
  AND avg_review_score IS NOT NULL;
