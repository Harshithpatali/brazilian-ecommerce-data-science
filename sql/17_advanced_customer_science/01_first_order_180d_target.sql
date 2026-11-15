-- First-order-only second-purchase target; no post-index predictors.
CREATE SCHEMA IF NOT EXISTS research;
CREATE OR REPLACE VIEW research.first_order_repeat_180d AS
WITH delivered AS (
 SELECT *, ROW_NUMBER() OVER(PARTITION BY customer_unique_id ORDER BY order_purchase_timestamp,order_id) rn
 FROM analytics.order_level
 WHERE order_status='delivered' AND customer_unique_id IS NOT NULL AND order_delivered_customer_date IS NOT NULL
), first_orders AS (SELECT * FROM delivered WHERE rn=1),
second_orders AS (SELECT customer_unique_id,MIN(order_purchase_timestamp) second_purchase FROM delivered WHERE rn=2 GROUP BY 1),
cutoff AS (SELECT MAX(order_purchase_timestamp) max_purchase FROM analytics.order_level)
SELECT f.customer_unique_id,f.order_id first_order_id,f.order_purchase_timestamp first_purchase_timestamp,
 f.gross_order_value first_order_value,f.item_count first_item_count,
 f.freight_revenue/NULLIF(f.gross_order_value,0) first_freight_share,
 f.delivery_days first_delivery_days,f.delivery_delay_days first_delivery_delay_days,
 f.avg_review_score first_review_score,
 EXTRACT(EPOCH FROM (s.second_purchase-f.order_purchase_timestamp))/86400.0 days_to_second_purchase,
 CASE WHEN s.second_purchase<=f.order_purchase_timestamp+INTERVAL '180 days' THEN 1 ELSE 0 END repeat_180d
FROM first_orders f CROSS JOIN cutoff c LEFT JOIN second_orders s USING(customer_unique_id)
WHERE f.order_purchase_timestamp+INTERVAL '180 days'<=c.max_purchase;
