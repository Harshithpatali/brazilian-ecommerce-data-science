CREATE OR REPLACE VIEW analytics.cohort_retention AS
WITH delivered AS (
    SELECT DISTINCT customer_unique_id,
           DATE_TRUNC('month', order_purchase_timestamp)::date AS order_month
    FROM analytics.order_level
    WHERE order_status = 'delivered'
),
first_purchase AS (
    SELECT customer_unique_id, MIN(order_month) AS cohort_month
    FROM delivered
    GROUP BY customer_unique_id
),
activity AS (
    SELECT
        d.customer_unique_id,
        f.cohort_month,
        d.order_month,
        (
            (EXTRACT(YEAR FROM d.order_month) - EXTRACT(YEAR FROM f.cohort_month)) * 12
            + EXTRACT(MONTH FROM d.order_month) - EXTRACT(MONTH FROM f.cohort_month)
        )::int AS months_since_first_order
    FROM delivered d
    JOIN first_purchase f USING (customer_unique_id)
),
cohort_size AS (
    SELECT cohort_month, COUNT(DISTINCT customer_unique_id) AS cohort_customers
    FROM first_purchase
    GROUP BY cohort_month
),
retained AS (
    SELECT
        cohort_month,
        months_since_first_order,
        COUNT(DISTINCT customer_unique_id) AS active_customers
    FROM activity
    GROUP BY cohort_month, months_since_first_order
)
SELECT
    r.cohort_month,
    r.months_since_first_order,
    r.active_customers,
    c.cohort_customers,
    r.active_customers::numeric / NULLIF(c.cohort_customers, 0) AS retention_rate
FROM retained r
JOIN cohort_size c USING (cohort_month)
ORDER BY 1, 2;
