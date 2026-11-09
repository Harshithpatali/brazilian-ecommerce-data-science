# Interview Guide — Brazilian E-Commerce Second-Purchase Intelligence

## 60-second project pitch

I built a point-in-time customer intelligence pipeline on the Brazilian Olist e-commerce dataset. The business problem is to identify customers likely to make a second delivered purchase within the next 90 days, so a retention or CRM team can prioritize limited campaign capacity.

The pipeline creates monthly customer snapshots, constructs features only from information observable before each snapshot, and trains interpretable and tree-based classifiers. It evaluates models with purged chronological validation rather than random splitting, compares them with recency/frequency baselines, calibrates probabilities using out-of-fold training predictions, and reports both ranking quality and top-K campaign outcomes. A Streamlit dashboard exposes model diagnostics and ranked historical prospects.

A key engineering decision: a strong cross-validation score is not enough. A candidate is marked research-only unless it beats the lifetime-frequency baseline on the untouched chronological holdout and at 5%, 10%, and 20% monthly campaign capacity.

## Business framing

- Decision: Which customers should receive a limited retention or second-purchase campaign?
- Eligible population: Customers after their first delivered order.
- Positive event: A second delivered purchase in [snapshot_date, snapshot_date + 90 days).
- Observation unit: One customer per monthly snapshot.
- Primary metric: Average precision / PR-AUC, because repeat purchase is a rare event.
- Operational metrics: Precision, recall, and lift at 5%, 10%, and 20% contact capacity, ranked within each monthly cohort.
- Guardrail: Compare against lifetime-frequency and other simple RFM baselines.

This is a second-purchase propensity task, not a conventional subscription churn model. SQL field churned=0 represents the positive repeat-purchase event.

## Architecture

    Olist source tables
      -> Analytics order-level layer
      -> Point-in-time SQL feature pipeline
      -> Customer-month snapshots and 90-day labels
      -> Data integrity and leakage checks
      -> Parquet training dataset
      -> Purged chronological CV
      -> Model and RFM baseline competition
      -> OOF probability calibration
      -> Validation threshold selection
      -> Untouched chronological holdout
      -> Promotion gate and versioned artifacts
      -> Streamlit customer intelligence dashboard

## Feature engineering

The current feature contract contains 57 point-in-time features, grouped into:

1. First-delivered-order profile: basket value, item/product/category/seller counts, freight share, installments, payment methods, delivery timing, and review information available before the snapshot.
2. Rolling 180-day behaviour: order frequency, basket composition, product diversity, category preference entropy, dominant category share, seller concentration (HHI and top-seller share), payment diversity, installment behaviour, credit-card share, and freight share.
3. Purchase-cycle behaviour: interpurchase gaps, gap variability, recency relative to expected purchase cycle, and recent-versus-prior frequency.
4. Monetary and engagement behaviour: rolling revenue, average order value, review score, delivery delay, active months, revenue per order, and recent revenue share.

Every feature must be based on purchases and deliveries observable before the snapshot. Reviews are available only after review_answer_timestamp. The SQL validator checks snapshot grain, target consistency, complete label windows, feature bounds, and review timing.

## Modeling and validation choices

- Candidate families: Logistic Regression, Random Forest, Extra Trees, HistGradientBoosting, XGBoost, CatBoost, and LightGBM, subject to installed dependencies.
- Baselines: recency-only, recent-frequency-only, lifetime-frequency-only, and dummy prior baseline.
- Splits: chronological train, validation, and final test periods. The 90-day target horizon is purged between periods so label windows do not overlap.
- CV: expanding-window folds grouped by snapshot date; rows from one date remain together.
- Calibration: Platt sigmoid fit using purged out-of-fold predictions from training only.
- Threshold: F2 is selected on the validation period, not on the final test.
- Final evaluation: PR-AUC, ROC-AUC, class-specific precision/recall/F1, F2, balanced accuracy, MCC, Brier score, log loss, calibration error, monthly top-K precision/recall/lift, known-versus-unseen customer metrics, and customer-clustered bootstrap uncertainty.

## How to discuss the model promotion gate

The best CV candidate is not automatically deployable. The final test period is used once to assess generalization. Promotion requires the candidate to beat lifetime frequency on holdout PR-AUC and precision at each predefined monthly contact capacity. If it fails, the pipeline can publish research artifacts, but metadata marks it research_only and the dashboard warns against production targeting.

**Latest v6 result (2026-11-06):** the database rebuild, integrity checks, training, model fingerprint validation, and artifact commit completed successfully. Random Forest was selected by temporal CV (PR-AUC 0.01196), but final holdout PR-AUC was 0.01048 versus 0.01665 for lifetime frequency. It also lost to that baseline at 5%, 10%, and 20% campaign capacity. Metadata correctly sets `pipeline_status=research_only`; the model must not be presented as production-ready. The key interview story is the evidence-based decision not to promote a model that failed its pre-registered baseline gate.

## Likely interview questions

### Why PR-AUC instead of accuracy?
The event is rare. A classifier can achieve high accuracy by predicting no repeat purchase for almost everyone. PR-AUC focuses on the positive event and makes precision/recall trade-offs visible. I compare PR-AUC with event prevalence and report top-K precision and lift.

### Why not random train/test split?
Customer snapshots are time-dependent and share customer histories. Random splitting can train on future periods and test on past periods, making estimates optimistic. Chronological splitting better represents deployment, while the 90-day purge prevents overlapping outcome windows.

### Why use customer_unique_id?
The Olist customer_id can represent an order-specific customer record. customer_unique_id is the stable identity needed to link repeat purchases across orders.

### How do you prevent leakage?
The SQL feature table filters purchase and delivery timestamps to be before the snapshot. Review scores require an answer timestamp before the snapshot. Target labels look forward 90 days, but future events are not included in features. The validator recomputes labels and checks point-in-time conditions.

### Why include simple baselines?
A complex model must prove it adds value over operationally cheap recency/frequency rules. If a model loses to lifetime frequency on the holdout, a business should not pay the complexity and maintenance cost.

### What does calibration add?
Ranking tells us who is more likely than whom. Calibration tests whether predicted probabilities correspond to observed frequencies, which matters when teams use probability thresholds, expected value, or budget forecasts.

### What is the limitation of the dataset?
Olist is historical, not a live CRM feed. Customer identity, campaign exposure, discounts, marketing contact history, and margin are not fully observed. This is propensity ranking, not causal uplift: a high-propensity customer is not necessarily someone whose purchase is caused by a campaign.

### What would you do next?
Validate on a later external time period or live shadow deployment; add campaign exposure and margin data; estimate incremental lift with randomized holdouts/uplift modeling; monitor calibration and population drift; and introduce retraining/version rollback controls.

## Demo checklist

1. Explain the decision and define the positive event before showing metrics.
2. Show the source-to-snapshot SQL and point-in-time review constraint.
3. Show the temporal split and embargo logic.
4. Compare candidate models against RFM baselines.
5. Show PR-AUC relative to prevalence and the monthly capacity table.
6. Explain the promotion gate and whether the current artifact is ready or research_only.
7. Open the dashboard and inspect the ranked latest historical snapshot.
8. State clearly that the Olist demo is historical, not a live customer feed.

## Reproducibility commands

    python -m pip install -r requirements.txt
    pytest -q
    python -m scripts.check_source_schema
    python -m scripts.profile_olist_data
    python -m scripts.refresh_ml_sql
    python -m scripts.validate_churn_data
    python -m scripts.run_ml
    streamlit run app/Home.py

Check the current Streamlit entrypoint if the home file has been renamed. Database credentials must be configured in environment variables or GitHub Actions secrets; never commit credentials.


## Customer intelligence extensions

### RFM segmentation
The dashboard derives recency from the analysis cutoff and last delivered purchase, frequency from delivered order count, and monetary value from customer revenue. Each dimension is assigned a relative quintile score. Segment labels are rule-based descriptions (for example, Champions or At Risk), not learned customer identities. Customer rows include a human-readable explanation of all three dimensions and a suggested interpretation.

### Time-to-second-purchase survival analysis
The Kaplan–Meier estimator models time from first to second delivered purchase. A customer with two or more delivered orders contributes an observed event at the interval between first and last delivered order. A one-order customer is right-censored at the latest date represented in the customer dataset. The survival curve estimates the probability of not yet observing a second purchase; one minus survival is the cumulative incidence estimate under non-informative censoring. Discuss censoring and the fact that this is historical observational data.

### Campaign scenario planner
The simulator allows an analyst to choose an audience, contact fraction, contact cost, assumed lift in percentage points, and contribution margin. It calculates contact volume, assumed incremental buyers, incremental revenue, gross contribution, campaign cost, net scenario contribution and break-even lift. The lift is an explicit input, not trained or identified from Olist. Olist has no randomized treatment/control assignment, so the simulator must never be described as causal uplift, measured ROI, or a validated forecast.

### Why the predictive model is research-only
The v6 Random Forest achieved holdout PR-AUC 0.01048 versus 0.01665 for lifetime-frequency ranking, with ROC-AUC 0.5494. It failed the promotion gate and is not deployed for production decisions. The correct next steps are feature ablation, leakage and cohort audits, stability analysis by time and customer tenure, and investigation of label sparsity/observation windows. Do not select features, thresholds, or models using the final test set. Keep the final holdout frozen until a new, pre-registered evaluation period is available.

### Suggested interview narrative
“I treated the predictive score as one component of a larger customer decision system. The model failed to outperform a simple lifetime-frequency baseline on the untouched holdout, so I did not ship it. I extended the product with descriptive RFM explanations, cohort analysis, survival analysis for purchase timing, and an assumption-led campaign economics simulator. This demonstrates that I can translate data into a business decision while communicating uncertainty and causal limitations.”
