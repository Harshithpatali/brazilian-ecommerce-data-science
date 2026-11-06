# Repeat-Purchase Propensity — Training and Evaluation

## Problem

At each monthly customer snapshot, rank active customers by the probability of an eventually-delivered purchase in the following 90 days. Eligible customers have at least one delivered order in the exact prior 180-day window and have reached their first delivered order.

The SQL label is \`churned\`: class 1 means no delivered purchase in the next 90 days; class 0 means a delivered purchase. Model scoring and ranking use \`1 - churned\` as the positive repeat-purchase event.

## Feature contract

There are 29 features: 27 numeric features and two categorical fields (\`customer_state\`, \`snapshot_month\`). Customer state is taken from the most recent delivered order observable before the snapshot. Snapshot month is derived from the snapshot date. Historical order, delivery and review timestamps must precede the snapshot.

## Candidate models

1. Logistic Regression — interpretable linear baseline.
2. HistGradientBoosting — nonlinear tabular benchmark.
3. Random Forest — bagged-tree benchmark.
4. CatBoost — gradient boosting with native categorical support for state and month.

Numeric imputation and one-hot encoding are fitted inside the train-only pipeline for scikit-learn candidates. CatBoost consumes the named categorical columns directly. No resampling or synthetic minority examples are created.

## Selection and calibration

- Unique snapshot dates are kept together throughout all splits.
- Train/validation/test partitions are chronological and separated by a 90-day target-window embargo.
- Expanding-window CV folds are built from unique snapshot dates and purge overlapping target windows.
- Each CV fit and validation fold must contain at least 15 repeat-purchase and 15 no-purchase examples; otherwise the fold is rejected.
- GridSearchCV selects the winning candidate by mean average precision for repeat purchase.
- A Platt sigmoid is fitted only on the separate chronological validation period. The final holdout does not select the candidate, calibration model, or campaign capacity.

## Operational policy

The primary operating policy ranks customers separately within each monthly snapshot and selects the top 10% by calibrated repeat-purchase probability. Top-5% and top-20% results are reported as alternative capacity scenarios. This is designed to give a campaign team an explicit contact-volume trade-off; it is not a claim that 10% is the economically optimal level.

## Evaluation outputs

The holdout report contains:

- Repeat-purchase PR-AUC and ROC-AUC.
- Prior-prevalence and recency-only baseline PR-AUC.
- Brier score, log loss, Brier skill score and calibration deciles.
- Precision, recall, F1, F2, balanced accuracy and Matthews correlation.
- Precision, recall, lift and flagged share at top 5%, 10% and 20% capacity per month.
- Customer-cluster bootstrap 95% PR-AUC confidence interval (500 resamples).
- Candidate model comparison, split dates, class counts, confusion matrix and prediction-level scores.

The final v5 holdout metrics are intentionally not documented as results until the full database rebuild, validation, training, metadata-contract verification and artifact commit succeed.

## Artifacts

The training run writes:

- \`models/best_churn_model.joblib\` — calibrated v5 propensity model.
- \`models/model_metadata.json\` — feature/target contract, population, calibration, splits, baselines, confidence interval and holdout metrics.
- \`models/model_comparison.csv\` — purged temporal-CV ranking and candidate parameters.
- \`models/test_predictions.parquet\` — final holdout probabilities and capacity flags.
- \`models/test_classification_report.json\` — class-level report.
- \`models/test_confusion_matrix.json\` — final holdout confusion matrix under top-10% capacity.
- \`models/test_precision_recall_curve.csv\` — final holdout precision-recall curve.
- \`models/test_calibration.csv\` — calibration decile table.
- \`data/ml/customer_churn_dataset.parquet\` — rebuilt customer-month model dataset.

## Production caveat

Offline propensity performance does not establish incremental campaign impact. Before using model scores to contact customers or issue incentives, run a randomized holdout or uplift experiment and evaluate incremental purchases, margin, cost per incremental buyer, ROI and customer-level guardrails. Live deployment additionally requires monitoring for data drift, calibration drift, delayed labels and changing campaign policies.
