CREATE OR REPLACE VIEW analytics.monthly_revenue_growth AS
WITH base AS (
    SELECT month, revenue
    FROM analytics.monthly_revenue
),
lagged AS (
    SELECT *,
           LAG(revenue) OVER (ORDER BY month) AS prior_month_revenue
    FROM base
)
SELECT *,
       revenue - prior_month_revenue AS revenue_change,
       (revenue - prior_month_revenue) / NULLIF(prior_month_revenue, 0) AS revenue_growth_rate
FROM lagged
ORDER BY month;
