# Model Card — Churn Baseline

## Intended use

Portfolio baseline for identifying customers who are currently one-time purchasers versus repeat purchasers.

## Important limitation

The current baseline target is not a true forward-looking churn label. It is included as an engineering baseline only. A production-quality version should create a point-in-time dataset:

```text
observation window -> feature window -> future prediction window
```

and define churn as no purchase during the future prediction window.

This prevents target leakage and makes the model truly predictive.

## Baseline

- Model: Logistic Regression
- Preprocessing: median imputation + standardization
- Class weighting: balanced
- Evaluation: ROC-AUC + classification report
