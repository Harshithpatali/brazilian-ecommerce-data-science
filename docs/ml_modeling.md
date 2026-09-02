# Churn Modeling

## Problem
Predict whether an active customer at a monthly snapshot will make **no delivered purchase during the following 90 days**.

## Leakage control
Features are calculated only from purchases before the snapshot date. The latest chronological snapshots are held out as the final test set.

## Feature engineering
Derived features include recent-order share, recent-revenue share, order/revenue velocity, product/seller diversity per order, missingness indicators, and log transforms for skewed behavioral variables.

## Models
Three model families are compared:

1. Logistic Regression — interpretable linear baseline.
2. Random Forest — nonlinear bagged tree ensemble.
3. HistGradientBoosting — boosted tree model suited to tabular numeric data.

## Tuning
Randomized hyperparameter search is performed with expanding-window temporal cross-validation. The primary optimization metric is **PR-AUC**, with ROC-AUC and F1 reported as secondary metrics.

## Saved artifacts
After `python -m scripts.run_ml`:

- `models/best_churn_model.joblib` — selected fitted pipeline.
- `models/model_comparison.csv` — model benchmark.
- `models/model_metadata.json` — features, hyperparameters, split dates and test metrics.
- `models/test_predictions.parquet` — final chronological test predictions.

The Streamlit app loads the saved best model directly. No FastAPI service is required.
