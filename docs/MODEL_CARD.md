# Model Card — Second-Purchase Propensity Intelligence

## Summary

| Field | Description |
|---|---|
| Project | Brazilian E-Commerce Customer Intelligence |
| Model artifact version | churn-pit-v6 |
| Task | Rank customers by probability of a second delivered purchase within 90 days |
| Unit | Customer-month snapshot |
| Positive event | A delivered purchase in the 90-day forward window |
| Intended users | CRM, retention, analytics, and data science teams |
| Interface | Streamlit dashboard |
| Training data | Historical Brazilian Olist marketplace dataset |

## Intended use

Use as a historical demonstration and research workflow for prioritizing customer retention analysis. The dashboard's ranked list illustrates monthly contact-capacity selection. It is not a live customer feed and should not trigger automated communications without independent validation and business approval.

## Target definition

Customers become eligible after their first delivered order. At a monthly snapshot date, the label is positive when a subsequent delivered purchase occurs in [snapshot_date, snapshot_date + 90 days).

The SQL dataset uses churned=0 for the repeat-purchase event and churned=1 for no delivered purchase in the horizon. Reports invert this label when computing repeat-purchase probabilities and metrics.

## Data and feature controls

- Stable identity: customer_unique_id.
- Snapshot grain: one row per customer and snapshot date.
- Feature availability: purchase and delivery timestamps must be before the snapshot.
- Review availability: review_answer_timestamp < snapshot_date.
- Label completeness: the full 90-day outcome window must be covered by source data.
- New v6 feature groups include basket/product diversity, category preference, seller concentration, and payment behaviour.
- SQL integrity checks cover duplicate keys, target mismatches, invalid values, incomplete labels, historical feature constraints, first-order profile constraints, and review timing.

## Evaluation protocol

- Chronological train, validation, and final holdout periods.
- 90-day purging between periods to avoid overlapping target windows.
- Expanding-window CV grouped by snapshot date.
- Candidate model competition against recency/frequency and dummy baselines.
- Platt calibration trained on purged out-of-fold training predictions.
- F2 threshold selected on validation data only.
- Final holdout reported using PR-AUC, ROC-AUC, threshold metrics, calibration metrics, campaign capacity metrics, known/unseen customer slices, and customer-clustered bootstrap intervals.

## Promotion policy

A candidate is eligible for production only if it beats the lifetime-frequency baseline on untouched holdout PR-AUC and precision at 5%, 10%, and 20% monthly campaign capacity. Otherwise metadata must say research_only. This policy is intentionally conservative and should be reviewed alongside business costs and expected campaign value.

## Known limitations and risks

- Historical Olist data does not represent a current production customer population.
- Campaign exposure, marketing costs, margin, and treatment assignment are not adequately represented; propensity is not uplift or causal impact.
- Customer snapshots are correlated; uncertainty estimates use customer-level resampling, but residual dataset bias remains possible.
- New category/seller/payment features may be sparse or sensitive to marketplace changes.
- Historical test performance is not a guarantee of future performance.
- Calibration can drift as customer behaviour and market conditions change.
- The model should not be used for credit, employment, or other high-impact decisions.

## Monitoring recommendations

For a real deployment, monitor:
- Input schema, missingness, and feature distributions.
- Monthly event prevalence and population stability.
- PR-AUC and top-K precision/recall when labels mature.
- Calibration/Brier score and predicted-to-observed event ratio.
- Campaign cost, incremental conversion, margin, and contact fatigue.
- Model and feature schema versions, artifact hash, retraining date, and rollback readiness.

## Current validation status

**Latest v6 validation (2026-11-06):** full SQL rebuild, point-in-time/target integrity validation, model training, metadata/model fingerprint checks, and artifact publication completed successfully. The selected Random Forest achieved holdout PR-AUC **0.01048** (ROC-AUC **0.5494**) against lifetime-frequency PR-AUC **0.01665**. Precision at 5%, 10%, and 20% capacity was **1.21%, 1.01%, and 0.87%**, respectively, versus **1.69%, 1.21%, and 0.88%** for lifetime frequency. The promotion gate correctly rejected it: `pipeline_status=research_only`, `production_eligible=false`. The workflow and artifact pipeline are validated; predictive performance is not sufficient for production use.
