-- Dashboard/reporting layer.
-- These views are intentionally simple and stable so Streamlit can consume them.
SELECT 'executive_kpis' AS dataset, COUNT(*) AS rows FROM analytics.executive_kpis
UNION ALL
SELECT 'monthly_revenue', COUNT(*) FROM analytics.monthly_revenue
UNION ALL
SELECT 'customer_metrics', COUNT(*) FROM analytics.customer_metrics
UNION ALL
SELECT 'customer_segments', COUNT(*) FROM analytics.customer_segments
UNION ALL
SELECT 'cohort_retention', COUNT(*) FROM analytics.cohort_retention
UNION ALL
SELECT 'category_performance', COUNT(*) FROM analytics.category_performance
UNION ALL
SELECT 'seller_performance', COUNT(*) FROM analytics.seller_performance
UNION ALL
SELECT 'delivery_performance', COUNT(*) FROM analytics.delivery_performance
UNION ALL
SELECT 'delivery_vs_review', COUNT(*) FROM analytics.delivery_vs_review
UNION ALL
SELECT 'geographic_performance', COUNT(*) FROM analytics.geographic_performance
ORDER BY 1;
