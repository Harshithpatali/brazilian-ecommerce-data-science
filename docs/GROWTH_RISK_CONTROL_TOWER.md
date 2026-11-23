# E-commerce Growth & Risk Control Tower

## Scope
Adds a dark operational dashboard, randomized synthetic campaign benchmark, T-learner uplift modeling, constrained price search from descriptive log-log elasticity, unsupervised transaction anomaly ranking and transparent rules-based intervention policy.

## Data integrity and interpretation
- Olist has no randomized marketing assignment. The campaign generator creates a separate reproducible synthetic experiment with known heterogeneous effects. Never describe its uplift as measured Olist campaign impact.
- Historical Olist prices and quantities are observational. The log-log coefficient is descriptive, not automatically causal. Causal price recommendations require randomized variation or a defensible quasi-experiment.
- Olist has no verified fraud labels. Isolation Forest scores are anomaly rankings, not fraud probabilities or confirmed fraud. Human review and adjudicated labels are required before automated blocking.
- The screenshot-inspired cancellations, returns, payment events and alerts are a synthetic UI demonstration, not actual Olist incidents.

## Models
- T-learner: separate treatment/control outcome models; uplift is predicted treatment response minus control response. Report randomized ITT alongside model-estimated ATE and top-k uplift.
- Pricing: fit a log-log quantity-price regression and search a price grid above unit cost. Report fit and observational limitations.
- Fraud: Isolation Forest ranks multivariate anomalies. Thresholds require precision/recall evaluation against adjudicated labels before production.
- Decision policy: high anomaly risk routes to manual review before automated offers; incentives require expected contribution to exceed intervention costs.

## Reproducibility
Run pytest -q. Demo data uses fixed seeds. The page appears as Growth & Risk Control Tower in the sidebar.

## Deployment
Existing Streamlit deployment and entrypoint are retained. No paid service or database migration is required for this demo extension.

## Campaign budget allocation and drift monitoring
The campaign optimizer ranks customers by estimated incremental contribution after contact and discount costs, then selects positive-value customers under a fixed budget. This is a model-based scenario, not measured profit. The monitoring utility calculates population stability index (PSI) for numeric features; compare a frozen reference period with a later production period, not shuffled rows from the same sample. Suggested interpretation bands are <0.10 stable, 0.10–0.25 monitor, and ≥0.25 investigate; thresholds are heuristics and should be calibrated to the feature and business process.
