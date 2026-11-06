/*
POINT-IN-TIME CUSTOMER CHURN DATASET

Target interval: [snapshot_date, snapshot_date + 90 days)
Features use only orders purchased before the snapshot and delivered to the
customer before the snapshot. Review scores are available only after
review_answer_timestamp. The source observation end uses all source orders.
Every eligible monthly snapshot is retained: 90-day churn is not permanent.
*/

DROP TABLE IF EXISTS ml.order_behavior CASCADE;
DROP TABLE IF EXISTS ml.first_delivered_orders CASCADE;
DROP TABLE IF EXISTS ml.delivered_orders CASCADE;
DROP TABLE IF EXISTS ml.customer_monthly_activity CASCADE;
DROP TABLE IF EXISTS ml.customer_snapshot_calendar CASCADE;
DROP TABLE IF EXISTS ml.customer_churn_dataset CASCADE;
DROP TABLE IF EXISTS ml.review_events CASCADE;

CREATE TABLE ml.delivered_orders AS
SELECT
    customer_unique_id,
    order_id,
    order_purchase_timestamp,
    order_delivered_customer_date,
    COALESCE(gross_order_value, 0)::numeric AS gross_order_value,
    delivery_delay_days
FROM analytics.order_level
WHERE order_status = 'delivered'
  AND customer_unique_id IS NOT NULL
  AND order_purchase_timestamp IS NOT NULL
  AND order_delivered_customer_date IS NOT NULL;

CREATE INDEX idx_ml_delivered_customer_purchase
    ON ml.delivered_orders (customer_unique_id, order_purchase_timestamp);
CREATE INDEX idx_ml_delivered_customer_delivery
    ON ml.delivered_orders (customer_unique_id, order_delivered_customer_date);
CREATE INDEX idx_ml_delivered_purchase
    ON ml.delivered_orders (order_purchase_timestamp);
ANALYZE ml.delivered_orders;

/*
ORDER-LEVEL CUSTOMER BEHAVIOUR
Pre-aggregate item and payment facts once at order grain. The customer-month
model only joins orders whose purchase and delivery timestamps precede the
snapshot, avoiding future item/payment leakage.
*/
CREATE TABLE ml.order_behavior AS
WITH item_features AS (
    SELECT
        oi.order_id,
        COUNT(*)::integer AS item_count,
        COUNT(DISTINCT oi.product_id)::integer AS distinct_products,
        COUNT(DISTINCT COALESCE(p.product_category_name, 'unknown'))::integer
            AS distinct_categories,
        COUNT(DISTINCT oi.seller_id)::integer AS seller_count,
        COALESCE(SUM(oi.price), 0)::numeric AS item_price,
        COALESCE(SUM(oi.freight_value), 0)::numeric AS freight_value
    FROM public.olist_order_items oi
    LEFT JOIN public.olist_products p
        ON p.product_id = oi.product_id
    GROUP BY oi.order_id
),
payment_features AS (
    SELECT
        order_id,
        COALESCE(MAX(payment_installments), 0)::integer AS max_installments,
        COUNT(DISTINCT payment_type)::integer AS payment_type_count,
        BOOL_OR(LOWER(payment_type) = 'credit_card')::integer
            AS is_credit_card,
        BOOL_OR(LOWER(payment_type) = 'boleto')::integer AS is_boleto,
        BOOL_OR(LOWER(payment_type) = 'debit_card')::integer
            AS is_debit_card,
        BOOL_OR(LOWER(payment_type) = 'voucher')::integer AS is_voucher
    FROM public.olist_order_payments
    GROUP BY order_id
)
SELECT
    ol.order_id,
    ol.customer_unique_id,
    ol.order_purchase_timestamp,
    ol.order_delivered_customer_date,
    COALESCE(ol.gross_order_value, 0)::numeric AS gross_order_value,
    COALESCE(i.item_count, ol.item_count, 0)::integer AS item_count,
    COALESCE(i.distinct_products, 0)::integer AS distinct_products,
    COALESCE(i.distinct_categories, 0)::integer AS distinct_categories,
    COALESCE(i.seller_count, 0)::integer AS seller_count,
    COALESCE(i.item_price, ol.product_revenue, 0)::numeric AS item_price,
    COALESCE(i.freight_value, ol.freight_revenue, 0)::numeric AS freight_value,
    COALESCE(
        i.freight_value / NULLIF(ol.gross_order_value, 0),
        0
    )::numeric AS freight_share,
    COALESCE(p.max_installments, 0)::integer AS max_installments,
    COALESCE(p.payment_type_count, 0)::integer AS payment_type_count,
    COALESCE(p.is_credit_card, 0)::integer AS is_credit_card,
    COALESCE(p.is_boleto, 0)::integer AS is_boleto,
    COALESCE(p.is_debit_card, 0)::integer AS is_debit_card,
    COALESCE(p.is_voucher, 0)::integer AS is_voucher,
    COALESCE(ol.delivery_days, 0)::numeric AS delivery_days,
    COALESCE(ol.delivery_delay_days, 0)::numeric AS delivery_delay_days,
    EXTRACT(EPOCH FROM (ol.order_approved_at - ol.order_purchase_timestamp))
        / 3600.0 AS approval_lag_hours
FROM analytics.order_level ol
LEFT JOIN item_features i ON i.order_id = ol.order_id
LEFT JOIN payment_features p ON p.order_id = ol.order_id
WHERE ol.customer_unique_id IS NOT NULL;

CREATE UNIQUE INDEX idx_ml_order_behavior_order
    ON ml.order_behavior (order_id);
CREATE INDEX idx_ml_order_behavior_customer_purchase
    ON ml.order_behavior (customer_unique_id, order_purchase_timestamp);
ANALYZE ml.order_behavior;

/* First delivered order establishes the second-purchase prediction cohort. */
CREATE TABLE ml.first_delivered_orders AS
SELECT customer_unique_id, order_id AS first_order_id,
       order_delivered_customer_date AS first_delivery_timestamp
FROM (
    SELECT
        customer_unique_id,
        order_id,
        order_delivered_customer_date,
        ROW_NUMBER() OVER (
            PARTITION BY customer_unique_id
            ORDER BY order_delivered_customer_date, order_id
        ) AS delivered_order_number
    FROM ml.delivered_orders
) ranked
WHERE delivered_order_number = 1;

CREATE UNIQUE INDEX idx_ml_first_delivered_customer
    ON ml.first_delivered_orders (customer_unique_id);
ANALYZE ml.first_delivered_orders;

/* Point-in-time review events support efficient as-of review features. */
CREATE TABLE ml.review_events AS
SELECT order_id, review_score, review_creation_date, review_answer_timestamp
FROM public.olist_order_reviews
WHERE order_id IS NOT NULL AND review_answer_timestamp IS NOT NULL;
CREATE INDEX idx_ml_review_order_date
    ON ml.review_events (order_id, review_answer_timestamp);
ANALYZE ml.review_events;

CREATE TABLE ml.customer_monthly_activity AS
SELECT
    customer_unique_id,
    DATE_TRUNC('month', order_purchase_timestamp)::date AS activity_month,
    COUNT(*) AS orders,
    SUM(gross_order_value) AS revenue,
    AVG(gross_order_value) AS avg_order_value,
    AVG(delivery_delay_days) AS avg_delivery_delay,
    MIN(order_purchase_timestamp) AS first_order_timestamp,
    MAX(order_purchase_timestamp) AS last_order_timestamp
FROM ml.delivered_orders
GROUP BY customer_unique_id, DATE_TRUNC('month', order_purchase_timestamp)::date;

CREATE INDEX idx_ml_customer_month
    ON ml.customer_monthly_activity (customer_unique_id, activity_month);
ANALYZE ml.customer_monthly_activity;

/* Begin after the first delivered order. This is a second-purchase propensity
   cohort: a customer becomes eligible after their first order is delivered.
   Only dates with a fully observed 90-day forward label window are materialized. */
CREATE TABLE ml.customer_snapshot_calendar AS
WITH observation_end AS (
    SELECT MAX(order_purchase_timestamp) AS max_observed_purchase_timestamp
    FROM analytics.order_level
    WHERE order_purchase_timestamp IS NOT NULL
)
SELECT
    s.customer_unique_id,
    generated.snapshot_timestamp::date AS snapshot_date
FROM ml.first_delivered_orders s
CROSS JOIN observation_end o
CROSS JOIN LATERAL GENERATE_SERIES(
    (
        DATE_TRUNC('month', s.first_delivery_timestamp)::date
        + INTERVAL '1 month'
    )::timestamp,
    DATE_TRUNC('month', o.max_observed_purchase_timestamp)::timestamp,
    INTERVAL '1 month'
) AS generated(snapshot_timestamp)
WHERE o.max_observed_purchase_timestamp IS NOT NULL
  AND generated.snapshot_timestamp + INTERVAL '90 days'
      <= o.max_observed_purchase_timestamp;

CREATE UNIQUE INDEX idx_ml_snapshot_customer_date
    ON ml.customer_snapshot_calendar (customer_unique_id, snapshot_date);
CREATE INDEX idx_ml_snapshot_date
    ON ml.customer_snapshot_calendar (snapshot_date);
ANALYZE ml.customer_snapshot_calendar;

/* Point-in-time features and order-timestamp-based forward target. */
CREATE TABLE ml.customer_churn_dataset AS
SELECT
    s.customer_unique_id,
    s.snapshot_date,
    first_order.gross_order_value AS first_order_gross_value,
    first_order.item_count AS first_order_item_count,
    first_order.distinct_products AS first_order_distinct_products,
    first_order.distinct_categories AS first_order_distinct_categories,
    first_order.seller_count AS first_order_seller_count,
    first_order.freight_share AS first_order_freight_share,
    first_order.max_installments AS first_order_max_installments,
    first_order.payment_type_count AS first_order_payment_type_count,
    first_order.is_credit_card AS first_order_is_credit_card,
    first_order.is_boleto AS first_order_is_boleto,
    first_order.is_debit_card AS first_order_is_debit_card,
    first_order.is_voucher AS first_order_is_voucher,
    first_order.delivery_days AS first_order_delivery_days,
    first_order.delivery_delay_days AS first_order_delivery_delay_days,
    first_order.approval_lag_hours AS first_order_approval_lag_hours,
    first_review.first_order_review_score,
    profile.avg_items_per_order_180d,
    profile.avg_categories_per_order_180d,
    profile.avg_sellers_per_order_180d,
    profile.avg_payment_installments_180d,
    profile.credit_card_order_share_180d,
    profile.avg_freight_share_180d,
    f.orders_180d,
    f.orders_90d,
    f.orders_30d,
    prior.orders_prior_180d,
    history.lifetime_delivered_orders,
    history.customer_tenure_days,
    history.positive_interpurchase_gap_count,
    history.zero_interpurchase_gap_share,
    history.avg_interpurchase_days,
    history.median_interpurchase_days,
    history.interpurchase_gap_cv,
    history.recency_days::numeric
        / NULLIF(history.avg_interpurchase_days, 0) AS recency_to_avg_interpurchase_ratio,
    f.orders_180d::numeric / NULLIF(prior.orders_prior_180d, 0)
        AS recent_vs_prior_order_frequency,
    f.revenue_180d,
    prior.revenue_prior_180d,
    f.revenue_90d,
    f.revenue_30d,
    f.avg_order_value_180d,
    review.avg_review_score_180d,
    f.avg_delivery_delay_180d,
    f.active_months_180d,
    f.last_order_date,
    f.recency_days,
    f.orders_90d::numeric / NULLIF(f.orders_180d, 0) AS recent_order_share,
    f.revenue_90d / NULLIF(f.revenue_180d, 0) AS recent_revenue_share,
    f.orders_180d::numeric / NULLIF(f.active_months_180d, 0)
        AS orders_per_active_month,
    f.revenue_180d / NULLIF(f.orders_180d, 0) AS revenue_per_order,
    CASE
        WHEN f.orders_180d > f.orders_90d
        THEN f.orders_90d::numeric
             / NULLIF(f.orders_180d - f.orders_90d, 0)
        ELSE NULL
    END AS recent_vs_older_order_ratio,
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM ml.delivered_orders future_order
            WHERE future_order.customer_unique_id = s.customer_unique_id
              AND future_order.order_purchase_timestamp >= s.snapshot_date
              AND future_order.order_purchase_timestamp
                    < s.snapshot_date + INTERVAL '90 days'
        ) THEN 0
        ELSE 1
    END AS churned
FROM ml.customer_snapshot_calendar s
JOIN ml.first_delivered_orders first_order_key
    ON first_order_key.customer_unique_id = s.customer_unique_id
JOIN ml.order_behavior first_order
    ON first_order.order_id = first_order_key.first_order_id
LEFT JOIN LATERAL (
    SELECT AVG(r.review_score)::numeric AS first_order_review_score
    FROM ml.review_events r
    WHERE r.order_id = first_order_key.first_order_id
      AND r.review_answer_timestamp < s.snapshot_date
) first_review ON TRUE
CROSS JOIN LATERAL (
    SELECT
        AVG(ob.item_count)::numeric AS avg_items_per_order_180d,
        AVG(ob.distinct_categories)::numeric AS avg_categories_per_order_180d,
        AVG(ob.seller_count)::numeric AS avg_sellers_per_order_180d,
        AVG(ob.max_installments)::numeric AS avg_payment_installments_180d,
        AVG(ob.is_credit_card::numeric) AS credit_card_order_share_180d,
        AVG(ob.freight_share)::numeric AS avg_freight_share_180d
    FROM ml.delivered_orders d
    JOIN ml.order_behavior ob ON ob.order_id = d.order_id
    WHERE d.customer_unique_id = s.customer_unique_id
      AND d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '180 days'
      AND d.order_purchase_timestamp < s.snapshot_date
      AND d.order_delivered_customer_date < s.snapshot_date
) profile
CROSS JOIN LATERAL (
    SELECT
        COUNT(*)::integer AS orders_180d,
        COUNT(*) FILTER (
            WHERE d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '90 days'
        )::integer AS orders_90d,
        COUNT(*) FILTER (
            WHERE d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '30 days'
        )::integer AS orders_30d,
        COALESCE(SUM(d.gross_order_value), 0)::numeric AS revenue_180d,
        COALESCE(
            SUM(d.gross_order_value) FILTER (
                WHERE d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '90 days'
            ), 0
        )::numeric AS revenue_90d,
        COALESCE(
            SUM(d.gross_order_value) FILTER (
                WHERE d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '30 days'
            ), 0
        )::numeric AS revenue_30d,
        AVG(d.gross_order_value)::numeric AS avg_order_value_180d,
        AVG(d.delivery_delay_days)::numeric AS avg_delivery_delay_180d,
        COUNT(DISTINCT DATE_TRUNC('month', d.order_purchase_timestamp))::integer
            AS active_months_180d,
        MAX(d.order_purchase_timestamp)::date AS last_order_date,
        (s.snapshot_date - MAX(d.order_purchase_timestamp)::date)::integer
            AS recency_days
    FROM ml.delivered_orders d
    WHERE d.customer_unique_id = s.customer_unique_id
      AND d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '180 days'
      AND d.order_purchase_timestamp < s.snapshot_date
      AND d.order_delivered_customer_date < s.snapshot_date
) f
LEFT JOIN LATERAL (
    /* The previous 180-day period is a separate point-in-time window.
       These features measure whether ordering/revenue is accelerating. */
    SELECT
        COUNT(*)::integer AS orders_prior_180d,
        COALESCE(SUM(d.gross_order_value), 0)::numeric AS revenue_prior_180d
    FROM ml.delivered_orders d
    WHERE d.customer_unique_id = s.customer_unique_id
      AND d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '360 days'
      AND d.order_purchase_timestamp < s.snapshot_date - INTERVAL '180 days'
      AND d.order_delivered_customer_date < s.snapshot_date
) prior ON TRUE
LEFT JOIN LATERAL (
    /* All history is still point-in-time: only purchases and deliveries
       completed before the snapshot are included. LAG derives order gaps. */
    SELECT
        COUNT(*)::integer AS lifetime_delivered_orders,
        (s.snapshot_date - MIN(history.first_purchase_date))::integer
            AS customer_tenure_days,
        COUNT(history.gap_days) FILTER (WHERE history.gap_days > 0)::integer
            AS positive_interpurchase_gap_count,
        COALESCE(
            COUNT(*) FILTER (WHERE history.gap_days = 0)::numeric
            / NULLIF(COUNT(history.gap_days), 0),
            0
        ) AS zero_interpurchase_gap_share,
        AVG(history.gap_days) FILTER (WHERE history.gap_days > 0)::numeric
            AS avg_interpurchase_days,
        percentile_cont(0.5) WITHIN GROUP (ORDER BY history.gap_days)
            FILTER (WHERE history.gap_days > 0)::numeric
            AS median_interpurchase_days,
        STDDEV_POP(history.gap_days) FILTER (WHERE history.gap_days > 0)::numeric
            / NULLIF(
                AVG(history.gap_days) FILTER (WHERE history.gap_days > 0),
                0
            ) AS interpurchase_gap_cv,
        (s.snapshot_date - MAX(history.purchase_date))::integer AS recency_days
    FROM (
        SELECT
            d.order_purchase_timestamp::date AS purchase_date,
            FIRST_VALUE(d.order_purchase_timestamp::date) OVER (
                ORDER BY d.order_purchase_timestamp, d.order_id
            ) AS first_purchase_date,
            EXTRACT(
                EPOCH FROM (
                    d.order_purchase_timestamp
                    - LAG(d.order_purchase_timestamp) OVER (
                        ORDER BY d.order_purchase_timestamp, d.order_id
                    )
                )
            ) / 86400.0 AS gap_days
        FROM ml.delivered_orders d
        WHERE d.customer_unique_id = s.customer_unique_id
          AND d.order_purchase_timestamp < s.snapshot_date
          AND d.order_delivered_customer_date < s.snapshot_date
    ) history
) history ON TRUE
LEFT JOIN LATERAL (
    /* Average reviews per order first, then weight each order equally.
       Multiple review rows for one order must not dominate this feature. */
    SELECT AVG(order_review.order_avg_review_score)::numeric
        AS avg_review_score_180d
    FROM ml.delivered_orders d
    CROSS JOIN LATERAL (
        SELECT AVG(r.review_score)::numeric AS order_avg_review_score
        FROM ml.review_events r
        WHERE r.order_id = d.order_id
          AND r.review_answer_timestamp < s.snapshot_date
    ) order_review
    WHERE d.customer_unique_id = s.customer_unique_id
      AND d.order_purchase_timestamp >= s.snapshot_date - INTERVAL '180 days'
      AND d.order_purchase_timestamp < s.snapshot_date
      AND d.order_delivered_customer_date < s.snapshot_date
      AND order_review.order_avg_review_score IS NOT NULL
) review ON TRUE
WHERE f.orders_180d > 0;

CREATE UNIQUE INDEX idx_ml_churn_customer_snapshot
    ON ml.customer_churn_dataset (customer_unique_id, snapshot_date);
CREATE INDEX idx_ml_churn_snapshot
    ON ml.customer_churn_dataset (snapshot_date);
CREATE INDEX idx_ml_churn_target
    ON ml.customer_churn_dataset (churned);
ANALYZE ml.customer_churn_dataset;
