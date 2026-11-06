# Churn Model Training and Evaluation

## Problem

Predict whether an eligible customer will have no eventually-delivered order
placed during the 90 days after a monthly point-in-time snapshot.

## Candidate models

1. Logistic Regression — interpretable linear baseline with balanced class weights.
2. HistGradientBoosting — nonlinear boosted-tree model for tabular data.
3. Random Forest — class-balanced, shallow/regularized bagged-tree candidate.

The current implementation compares all three candidates.

## Class imbalance and scoring objective

The rebuilt dataset contains 10,570 customer-month snapshots. Only 306 (2.89%)
have a delivered purchase in the next 90 days; 10,264 (97.11%) do not.

The estimator is fitted to the SQL target \`churned\`, where 1 means no delivered
purchase. To avoid optimizing for this near-constant majority class, temporal
CV average precision is computed for the minority repeat-purchase event
(\`1 - churned\`). The model retains its churn-probability output, and
repeat-purchase probability is its complement.

## Training and selection

- Unique snapshot dates remain intact when constructing temporal folds.
- Expanding-window cross-validation runs on the training period with a 90-day
  purge between each fold's training and validation dates.
- GridSearchCV optimizes average precision for repeat purchase within 90 days.
- The winning model and hyperparameters are selected using CV PR-AUC only.
  Final holdout metrics are never used to choose a candidate.

## Validation and final holdout

Unique dates are separated chronologically into train, validation, and final
test periods. A 90-day embargo is applied between each period because the
target looks 90 days into the future. Snapshots whose outcome windows overlap
the next period are excluded from the earlier partition.

- Validation selects the repeat-purchase probability threshold by maximum F2 (beta=2), deliberately favoring recall. The complementary churn-probability cutoff is stored for risk scoring.
- The final test period is evaluated once after the model and threshold are frozen.
- Test PR-AUC, ROC-AUC and F1 focus on the minority repeat-purchase event and
  are compared with a DummyClassifier prior-probability baseline.
- Reports also include both classes' precision/recall/F1, balanced accuracy,
  and Matthews correlation coefficient.
- The F2 threshold is not business-optimal without an intervention cost model and false-positive costs.

## Outputs

Running \`python -m scripts.run_ml\` writes:
- \`models/best_churn_model.joblib\` — selected fitted pipeline;
- \`models/model_comparison.csv\` — repeat-purchase CV PR-AUC and candidate parameters;
- \`models/model_metadata.json\` — version, periods, threshold, class distributions,
  baseline and final test metrics;
- \`models/test_predictions.parquet\` — final test predictions;
- \`models/test_classification_report.json\` — class-level report;
- \`models/test_confusion_matrix.json\` — confusion matrix;
- \`models/test_precision_recall_curve.csv\` — repeat-purchase precision-recall curve.

The dashboard refuses to serve older artifacts until metadata confirms the
corrected pipeline version has been trained.
