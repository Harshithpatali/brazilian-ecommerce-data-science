# Churn Model Training and Evaluation

## Problem

Predict whether an eligible customer will have no eventually-delivered order
placed during the 90 days after a monthly point-in-time snapshot.

## Candidate models

1. Logistic Regression — interpretable linear baseline with balanced class weights.
2. HistGradientBoosting — nonlinear boosted-tree model for tabular data.

The current implementation compares these two models. Random Forest is not
part of the training code.

## Training and selection

- Unique snapshot dates remain intact when constructing temporal folds.
- Expanding-window cross-validation is run on the training period with a 90-day
  purge between each fold's training and validation dates.
- GridSearchCV optimizes average precision (PR-AUC).
- The winning model and hyperparameters are selected using CV PR-AUC only.
  Final holdout metrics are never used to choose a candidate.

## Validation and final holdout

Unique dates are separated chronologically into train, validation, and final
test periods. A 90-day embargo is applied between each period because the
target looks 90 days into the future. Snapshots whose outcome windows overlap
the next period are excluded from the earlier partition.

- Validation selects a classification threshold by maximum F1.
- The final test period is evaluated once after the model and threshold are frozen.
- The selected model is compared with a DummyClassifier prior-probability baseline. Reports include per-class precision/recall/F1, balanced accuracy, and Matthews correlation coefficient so minority-class behavior is visible.
- The F1 threshold is not business-optimal without an intervention cost model.

## Outputs

Running python -m scripts.run_ml writes:
- models/best_churn_model.joblib — selected fitted pipeline;
- models/model_comparison.csv — CV PR-AUC and candidate hyperparameters;
- models/model_metadata.json — version, periods, threshold, distributions,
  baseline, and final test metrics;
- models/test_predictions.parquet — final test predictions;
- models/test_classification_report.json — class-level report;
- models/test_confusion_matrix.json — confusion matrix;
- models/test_precision_recall_curve.csv — test precision-recall curve.

The dashboard refuses to serve older artifacts until metadata confirms the
corrected pipeline version has been trained.
