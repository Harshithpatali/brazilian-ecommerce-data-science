# Second-purchase model: next experiment protocol

## Current evidence

The frozen v6 evaluation is not production eligible. On the previously reported final chronological holdout, the multivariate model achieved PR-AUC **0.01048**, below the lifetime-frequency baseline at **0.01665**. Precision at 5%, 10% and 20% contact capacity also failed the promotion gate. The existing holdout is a historical benchmark; it must not be used repeatedly to tune features or thresholds.

## What changed in this iteration

- Candidate pipelines can now be built against an explicit feature subset using `build_models(feature_cols=...)`.
- `scripts/run_repeat_purchase_ablation.py` compares interpretable feature families against a lifetime-frequency-only control.
- Every ablation uses the original training period only and purged, date-grouped, expanding-window CV.
- Validation and final-test rows are not read by the ablation scoring loop.
- The ablation report is written to `reports/model_research/repeat_purchase_ablation_cv.csv`.

## Run

```bash
python scripts/run_repeat_purchase_ablation.py
# Optional additional tree candidate when installed:
python scripts/run_repeat_purchase_ablation.py --models logistic_regression hist_gradient_boosting xgboost
pytest -q tests/test_temporal_splits.py tests/test_repeat_purchase_ablation.py
```

The script requires the rebuilt `data/ml/customer_churn_dataset.parquet` produced by the existing SQL pipeline. It does not download or replace source data.

## Feature hypotheses

| Group | Hypothesis |
|---|---|
| First-order profile + lifetime frequency | First-order basket, payment, delivery and review characteristics add signal beyond order count |
| Recency/frequency + lifetime frequency | Recent activity and purchase intervals improve on lifetime count |
| Transaction diversity + lifetime frequency | Product/category/seller/payment diversity adds incremental ranking value |
| Service/review + lifetime frequency | Delivery experience and review signals add incremental value |
| Revenue/value + lifetime frequency | Spend and order-value history adds incremental value |
| Full feature set | Combined features improve consistently across temporal folds |

## Promotion standard

1. Prefer a simpler candidate unless a richer feature family improves average precision consistently across purged folds.
2. Review fold-level scores and event support; do not select based on one favorable month.
3. Keep probability calibration and campaign threshold selection separate from ranking evaluation.
4. Do not promote from the previously inspected v6 holdout. A candidate selected after this investigation requires a **new, later, untouched temporal evaluation period**.
5. Require the candidate to beat lifetime frequency on PR-AUC and predefined monthly contact-capacity precision, with uncertainty intervals and subgroup diagnostics.
6. If a new future holdout cannot be obtained, retain `research_only` status. Do not state that the model beats the baseline.

## Business interpretation

The task is to rank first-time delivered customers by the likelihood of a second delivered purchase within 90 days. The operational decision is who to prioritize for a retention contact under a fixed monthly capacity. A higher PR-AUC alone does not establish incremental campaign profit; a randomized campaign or defensible causal evaluation is needed to estimate incremental impact.

## Known constraints

- Olist is a historical, observational marketplace dataset; it does not contain randomized CRM treatment assignment.
- Repeated customer snapshots are dependent; temporal folds and customer-cluster uncertainty are important.
- Customers without a fully observed 90-day outcome window must remain excluded/right-censored according to the SQL target contract.
- A model that fails the baseline comparison is still a valid research result, not a reason to tune against the final test.
