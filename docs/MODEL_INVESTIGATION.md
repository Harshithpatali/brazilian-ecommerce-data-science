# v6 predictive-model investigation and next experiments

## Frozen result (do not tune against it)

| Metric | Random Forest | Lifetime-frequency baseline |
|---|---:|---:|
| Temporal CV PR-AUC | 0.01196 | 0.01107 |
| Untouched holdout PR-AUC | 0.01048 | 0.01665 |
| Holdout precision @ 5% capacity | 1.21% | 1.69% |
| Holdout precision @ 10% capacity | 1.01% | 1.21% |
| Holdout precision @ 20% capacity | 0.87% | 0.88% |

Holdout ROC-AUC for the selected model is 0.5494. The positive event prevalence in the final test is approximately 0.616%. The candidate fails the promotion gate and remains research-only.

## What the evidence suggests

1. **Sparse event:** only 882 positive events occur among 143,208 final test snapshots. PR-AUC is therefore the primary ranking metric; accuracy is not informative.
2. **Simple behavioural signal dominates:** lifetime order frequency is stronger on the final holdout than the selected multivariate model. The model's CV advantage did not transfer.
3. **Generalization gap:** Random Forest leads lifetime frequency in temporal CV by about 0.00089 PR-AUC but trails it on holdout by about 0.00617. This is consistent with unstable weak signal, cohort drift, or model selection noise—not evidence that the holdout should be adjusted.
4. **Very few chronological test cohorts:** the holdout covers four monthly cohorts. Report month-level variation and uncertainty; do not treat a single pooled number as stable performance.
5. **Customer overlap:** the test contains both returning/previously observed and unseen customer identities. Report these cohorts separately and inspect their prevalence and ranking performance before deciding whether the use case should be known-customer CRM, new-customer activation, or both.
6. **Target interpretation:** the event is a second delivered purchase within 90 days after a first delivered order. It is not generic churn, subscription cancellation, or customer lifetime value.

## Safe investigation sequence

- Freeze v6 metadata, test predictions, thresholds, and holdout labels as an immutable benchmark.
- Add a training-only feature-ablation study: first-order profile only; recency/frequency only; transaction diversity only; delivery/review only; full features. Recompute every fold inside the temporal CV loop.
- Compare a regularized logistic model, shallow boosted trees, and a frequency-only score using identical temporal folds. Keep all hyperparameter selection inside training folds.
- Report fold-level and month-level PR-AUC, precision@K, recall@K, calibration, and confidence intervals. Use customer-cluster bootstrap where repeated snapshots create dependence.
- Audit event construction and label observability: first delivered purchase definition, exact second delivered purchase timestamp, 90-day horizon, right-censoring at data end, and 90-day purge boundaries.
- Audit feature availability for each snapshot, especially review-answer timestamps and delivery completion. No feature may use an event after the snapshot.
- Inspect performance by first-order month, customer tenure, order value bands, geography and known/unseen customer cohort. Treat subgroup results as diagnostics, not a new way to select the final test.
- If a new model is selected, evaluate it once on a newly collected future period after the design and threshold are frozen.

## Decision

Do not promote or use v6 to target real customers. Keep the lifetime-frequency benchmark visible. The next model iteration should focus on diagnosing whether any stable incremental signal exists beyond frequency, not on maximizing a metric through repeated test-set iteration.
