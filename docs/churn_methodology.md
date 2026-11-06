# Repeat-Purchase Propensity Methodology — v5

## Prediction question

At each monthly scoring date, among customers with at least one delivered order in the prior 180 days, which customers are most likely to place another order that is eventually delivered during the next 90 days?

This is a recurring CRM propensity-ranking problem. The population includes customers after their first delivered order; customers do not need to have repeated before becoming eligible.

## Snapshot and eligible population

- **Snapshot grain:** one row per customer per calendar month.
- **Entry:** the first snapshot is the first day of the month after the customer's first delivered order.
- **Eligibility:** the customer must have at least one order both purchased and delivered during the exact prior 180-day feature window.
- **Repeat snapshots:** customers can reappear in later months, including after a period without a purchase. Each row is a new decision moment.
- **Observation end:** a snapshot is materialized only if the complete next-90-day label window is observable.
- **Customer identity:** \`customer_unique_id\` is used for longitudinal customer grouping, not the per-order \`customer_id\`.

A read-only Supabase preflight estimated 357,413 customer-month decisions across 79,594 customers and 2,686 repeat-purchase decision rows (0.75%). These are preflight counts, not the final v5 materialized dataset or final holdout counts.

## Target definition

The target uses the half-open interval [snapshot_date, snapshot_date + 90 days).

- \`churned = 0\` when at least one eventually-delivered order has a purchase timestamp in this interval.
- \`churned = 1\` when no such order exists.
- A purchase exactly at the 90-day upper boundary is outside the interval.
- Eventual delivery status is used retrospectively to define the outcome; it is never a predictor.
- In model evaluation, repeat purchase (\`1 - churned\`) is the positive class.

The next-90-day label window must be complete before a snapshot is included.

## Point-in-time features

For snapshot date t, current activity features use delivered orders purchased in [t - 180 days, t) and delivered before t. Nested order and revenue windows cover the latest 90 and 30 days. A prior period [t - 360 days, t - 180 days) supports activity-trend ratios.

Lifetime features include delivered-order count, customer tenure, positive interpurchase-gap count, zero-gap share, positive-gap mean/median and gap variability. If orders share an exact purchase timestamp, that is represented as a zero-gap event; no artificial positive interval is added.

Customer state is taken from the **most recently purchased delivered order observable before t**. Snapshot month is derived from t. Review scores enter only when the review-answer timestamp precedes t. Undefined numerical features remain missing and are imputed inside each fitted pipeline, not by using information from a later snapshot.

All feature SQL is constrained by purchase and delivery timestamps earlier than t. Future records are used only to construct the label.

## Temporal evaluation protocol

Unique snapshot dates are split chronologically into training, a separate validation period and a final test period. A 90-day embargo is applied at each boundary because the target looks 90 days into the future. Training snapshots are retained only when their target window ends by the first validation date. Validation snapshots are retained only when their target window ends by the first test date.

Expanding-window CV folds are formed from unique snapshot dates, never individual rows. All customers from one snapshot month remain on the same side of a fold. Each fold purges training snapshots whose 90-day label windows overlap the validation block. The v5 training search requires at least 15 repeat-purchase and 15 non-repeat-purchase examples in each fit and validation block. When the available history cannot support this, training fails instead of silently using a random split or a near-empty positive fold.

Customers may occur in both earlier and later time periods. This is intentional for the use case of monthly scoring of known customers. It does **not** measure generalization to entirely new customers. Confidence intervals use customer-cluster bootstrap resampling so repeated monthly rows for one customer are not treated as independent observations.

## Model selection and calibration

Candidate families are Logistic Regression, HistGradientBoosting, Random Forest and CatBoost. The first three use a train-fitted numeric/categorical preprocessing pipeline; CatBoost receives customer state and snapshot month as native categorical features.

The winning candidate is selected using mean purged temporal-CV average precision for repeat purchase. A Platt sigmoid is then fitted on the separate chronological validation period to calibrate repeat-purchase probabilities. The final test period is not used for candidate selection, calibration or capacity-policy selection.

## Operational policy and reporting

The operational decision is capacity-based, not a global probability cutoff. For each monthly snapshot, customers are ranked by calibrated repeat-purchase probability and the top 5%, 10% or 20% are selected for an illustrative campaign list.

The final holdout reports:

- PR-AUC and ROC-AUC for repeat purchase, plus Gini and Kolmogorov–Smirnov (KS) separation.
- A training-prevalence prior baseline and a recency-only ranking baseline.
- Brier score, log loss, Brier skill score, expected calibration error and probability-calibration deciles.
- Precision, recall, F1, F2, balanced accuracy and Matthews correlation under the capacity policy.
- Precision, recall, lift and contacted share at top 5%, 10% and 20% monthly capacity.
- 95% customer-cluster bootstrap intervals for PR-AUC and top-10% campaign precision, recall and lift.
- Class distributions, split dates, model comparison, holdout predictions and confusion matrix.

PR-AUC must be interpreted relative to repeat-purchase prevalence. A ranking model should also be compared with the simple recency baseline; beating a constant prior alone is not enough evidence of useful customer ranking.

## Limitations and production requirements

- Olist is a historical dataset, not a live customer-event feed.
- Eventual delivery is part of the retrospective label definition.
- Historical performance on known customers does not establish performance on newly acquired customers.
- A propensity model predicts who is likely to purchase; it does not estimate who will purchase **because** they receive an offer.
- Before operational deployment, evaluate incremental effect using randomized campaign holdouts or an uplift design, and include contact cost, incentive cost, margin and campaign capacity.
- Offline ranking and calibration metrics are not a substitute for live monitoring, data-drift checks, calibration refresh, rollback and privacy/consent controls.
