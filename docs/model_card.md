# Model Card — Repeat-Purchase Propensity v5

## Status

Version: \`churn-pit-v5\`.

The v5 candidate pipeline is designed for monthly customer CRM ranking across customers after their first delivered purchase. Final holdout metrics will be recorded here only after the v5 GitHub Actions rebuild, integrity validation, training, fitted-model hash verification and artifact commit complete.

## Intended use

Offline analysis and demonstration of monthly repeat-purchase propensity ranking for historical Olist customers. The intended decision is to prioritize a limited fraction of active customers for possible CRM contact.

This model is not a live feed, a production-ready offer policy or a causal uplift estimator. It must not be used to deny service or make consequential decisions.

## Population and target

One row represents a customer-month decision. Customers become eligible after their first delivered order and must have at least one order purchased and delivered within the prior 180 days.

\`churned = 0\` means an eventually-delivered order has a purchase timestamp in the next 90 days. \`churned = 1\` means no such order exists. The target window is half-open: [snapshot_date, snapshot_date + 90 days).

The population can contain repeated rows for the same customer across different months. Uncertainty intervals therefore use customer-cluster bootstrap resampling.

A read-only preflight of the expanded first-delivery population found 357,413 customer-month decisions across 79,594 customers, including 2,686 repeat-purchase decisions (0.75%). These are preliminary preflight counts, not final v5 artifact counts.

## Features and leakage control

The 29-feature contract contains 27 numeric behavioral/value features and categorical customer state and snapshot month. Customer state is taken from the most recent delivered order observable before the snapshot. Purchase and delivery dates used as historical features are strictly before the snapshot. Review score is included only when its answer timestamp precedes the snapshot. Future orders are used only to form the target.

## Model candidates and evaluation

Candidate models: Logistic Regression, HistGradientBoosting, Random Forest and CatBoost. Model selection uses expanding-window date-grouped temporal CV, purged by the 90-day target horizon. Every fit and validation fold must contain at least 15 repeat-purchase and 15 no-purchase examples.

The selected model is calibrated with a Platt sigmoid fitted on the separate chronological validation period. The final chronological holdout is reserved for evaluation only.

The operational benchmark is top-10% monthly capacity. Top-5% and top-20% results are also recorded. Evaluation includes PR-AUC, ROC-AUC, prior-prevalence and recency baselines, Brier score, log loss, Brier skill score, calibration deciles, precision, recall, F1, F2, balanced accuracy, Matthews correlation, capacity precision/recall/lift, confusion matrix and a customer-cluster bootstrap 95% PR-AUC interval.

## Limitations

- Olist is a historical marketplace dataset, not a live stream.
- The model ranks purchase propensity; it does not estimate incremental treatment effect.
- Known-customer temporal evaluation does not establish unseen-customer generalization.
- Customer-month rows are repeated observations; confidence intervals must preserve customer clusters.
- A campaign decision requires controlled experiments, margin and incentive economics, contact-cost constraints, customer consent and operational monitoring.
- If the model does not improve materially on the recency-only benchmark or fails probability calibration, it should not be presented as a useful targeting model.
