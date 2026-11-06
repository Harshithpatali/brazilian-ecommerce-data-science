# Second-Purchase Propensity — Model Training and Evaluation

## Problem definition

After a customer's first delivered order, predict whether a second delivered
purchase will occur in the 90 days after each eligible monthly snapshot.

The source SQL target is \`churned\`: 0 means second delivered purchase within
90 days; 1 means no such purchase. The modeling and reporting positive class
is \`1 - churned\`.

## Candidate models

1. Logistic Regression — interpretable linear candidate with balanced class weights.
2. HistGradientBoosting — nonlinear boosted-tree candidate for tabular data.
3. Random Forest — class-balanced bagged-tree candidate.

They compete against three point-in-time univariate RFM scorers:
recency-only, recent-order-frequency-only and lifetime-order-frequency-only.
All six candidates use the same purged temporal-CV folds and average-precision
scorer. A complex model is selected only when it improves on the simple
behavioral rankings during cross-validation. Validation and final-test scores
are not used to choose the model family or hyperparameters.

## Data population

The connected Olist source audit found 99,441 source orders, 96,470 delivered
orders and 93,350 delivered customers. The first-delivered cohort query found
357,413 eligible customer-month snapshots for 79,594 customers and 2,686
second-purchase events (0.752%). These are source/cohort audit counts, not a
substitute for final pipeline or holdout metrics.

The training pipeline fails fast when it receives fewer than 300,000 rows,
70,000 unique customers or 2,000 positive second-purchase events. This
prevents the earlier, much smaller third-or-later-purchase cohort from being
mistaken for the second-purchase training population.

## Temporal model selection

- Snapshots are split by unique calendar date; same-date customers never
  straddle a partition or fold.
- Train, validation and final test are chronological with a 90-day embargo
  between adjacent periods.
- Expanding-window cross-validation uses 90-day purging so no training
  label window overlaps its fold's validation block.
- GridSearchCV scores \`P(second purchase within 90 days)\` using average
  precision (PR-AUC).
- Prior-probability, recency-only, recent-order-frequency-only and
  lifetime-order-frequency-only baselines are scored on the same final holdout.

## Probability calibration

A Platt sigmoid is fitted from purged out-of-fold training predictions only.
It is not fitted on the validation period or final test period. The validation
period is used to select a diagnostic F2 operating threshold. The calibrated
model and threshold are frozen before final-test evaluation.

Calibration metrics on the final test include Brier score, log-loss,
expected calibration error and reliability-bin summaries. Calibration is
reported separately from discrimination: a calibrated model can still have
weak ranking power.

## Campaign-capacity evaluation

For each monthly test cohort, customers are ranked by calibrated
second-purchase probability. The evaluation reports:
- Precision@K and recall@K at top 5%, 10% and 20% monthly contact capacity;
- number of customers contacted and repeat purchases captured;
- lift over the cohort's observed repeat-purchase prevalence;
- head-to-head precision at each contact capacity versus recency, recent
  frequency and lifetime frequency baselines.

These metrics answer the practical question: *given a fixed monthly outreach
budget, how many future second purchases occur among the customers selected?*
They are offline ranking estimates, not incremental lift caused by a campaign.

## Generalization and uncertainty

- A known-customer future test result is reported.
- Customers absent from both training and validation are isolated within the
  chronological test and evaluated separately.
- A customer-cluster bootstrap resamples customers, retaining each sampled
  customer's monthly rows together, to estimate a 95% PR-AUC interval.
  This accounts for within-customer dependence better than treating each
  customer-month as independent.

## Output artifacts

The v5 pipeline writes:
- \`models/best_churn_model.joblib\` — selected Platt-calibrated model wrapper;
- \`models/model_comparison.csv\` — temporal-CV PR-AUC and parameters;
- \`models/model_metadata.json\` — cohort definition, partitions, calibration,
  test metrics, customer overlap, capacity metrics, baselines and CI;
- \`models/test_predictions.parquet\` — final holdout predictions;
- \`models/test_classification_report.json\` and
  \`models/test_confusion_matrix.json\` — thresholded final-test report;
- \`models/test_precision_recall_curve.csv\` — threshold-independent ranking curve;
- \`models/test_capacity_metrics.json\` — top-5/10/20% monthly campaign metrics;
- \`models/test_calibration_bins.csv\` — reliability-bin data.

## Interpretation limits

Strong offline prediction does not prove a retention campaign will cause
additional orders. Incremental campaign lift requires randomized treatment
and control groups. Thresholds must reflect the real contact budget,
intervention cost and minimum acceptable precision; the final holdout must
remain reserved for the final unbiased report.
