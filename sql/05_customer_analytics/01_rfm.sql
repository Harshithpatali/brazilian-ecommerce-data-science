CREATE OR REPLACE VIEW analytics.customer_rfm AS
WITH base AS (
    SELECT
        customer_unique_id,
        MAX(order_purchase_timestamp)::date AS last_order_date,
        COUNT(*) FILTER (WHERE order_status = 'delivered') AS frequency,
        SUM(gross_order_value) FILTER (WHERE order_status = 'delivered') AS monetary
    FROM analytics.order_level
    WHERE order_status = 'delivered'
    GROUP BY customer_unique_id
),
scored AS (
    SELECT *,
        NTILE(5) OVER (ORDER BY (SELECT NULL), last_order_date) AS dummy,
        NTILE(5) OVER (ORDER BY last_order_date ASC) AS r_score_raw,
        NTILE(5) OVER (ORDER BY frequency) AS f_score,
        NTILE(5) OVER (ORDER BY monetary) AS m_score
    FROM base
),
final AS (
    SELECT
        customer_unique_id,
        last_order_date,
        frequency,
        monetary,
        6 - r_score_raw AS r_score,
        f_score,
        m_score
    FROM scored
)
SELECT *,
       (r_score + f_score + m_score) AS rfm_score,
       CASE
           WHEN r_score >= 4 AND f_score >= 4 AND m_score >= 4 THEN 'Champions'
           WHEN f_score >= 4 AND m_score >= 3 THEN 'Loyal Customers'
           WHEN r_score >= 4 AND f_score <= 2 THEN 'New / Potential'
           WHEN r_score <= 2 AND f_score >= 3 THEN 'At Risk'
           WHEN r_score <= 2 AND f_score <= 2 THEN 'Lost'
           ELSE 'Needs Attention'
       END AS segment
FROM final;
