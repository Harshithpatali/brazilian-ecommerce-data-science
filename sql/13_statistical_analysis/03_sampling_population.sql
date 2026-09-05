CREATE OR REPLACE VIEW analytics.sampling_population AS
SELECT
    customer_unique_id,
    segment,
    frequency,
    monetary,
    r_score,
    f_score,
    m_score,
    rfm_score
FROM analytics.customer_rfm
WHERE frequency IS NOT NULL
  AND monetary IS NOT NULL
  AND segment IS NOT NULL;
