# Advanced Customer Science Extensions

## Status and scope
This commit adds reusable methods and an additional Streamlit page. It does not claim that database-dependent models have been fitted or that Olist contains randomized campaign exposure. The existing SQL pipeline, Kaplan-Meier view and 90-day model remain intact.

## BG/NBD and Gamma-Gamma
Install the optional lifetimes package and prepare delivered-order transactions with customer_unique_id, date and monetary_value. Fit BG/NBD on a calibration window, then compare predictions with observed purchases by frequency bucket in a later holdout. Gamma-Gamma is fitted only on repeaters; report their count and parameter instability. Sparse repeat history is a limitation, not a reason to suppress the result.

## First-order 180-day prediction
Run sql/17_advanced_customer_science/01_first_order_180d_target.sql after the existing SQL build. The view uses stable customer identity, first delivered-order features and a fully observable 180-day outcome window. Build a separate logistic/boosting benchmark, evaluate PR-AUC relative to prevalence, calibration and top-decile lift. Never mix post-index orders or reviews into predictors. Preserve the existing monthly 90-day model as a separate task.

## Survival
Existing Kaplan-Meier analysis remains. Fit Cox PH with duration and event from the customer survival table, inspect proportional-hazards diagnostics, and use stratification or Weibull AFT if needed. Cure-model interpretation is a sensitivity discussion: standard KM/Cox does not identify a permanent never-returner fraction without stronger assumptions.

## Observational delivery-delay inference
The new crossfit_dr_effect utility implements cross-fitted AIPW using a logistic propensity model and random-forest outcome regressions. It reports ATE, standard error, confidence interval, propensity overlap and pre-treatment SMD. This is not a causal forest or a guarantee against confounding. Treatment should be late delivery; outcome should be second delivered purchase after the index delivery. Include category, value, freight, seller/customer state, purchase month and product size/weight as pre-treatment covariates. Do not control for review score when estimating the total effect because it may be a mediator. Report weighted balance, effective sample size and sensitivity analysis. Interpret causally only under consistency, positivity and no unmeasured confounding.

## Experimentation
The app provides a two-proportion sample-size calculator, sequential-look guard, and the Python utilities include CUPED adjustment and bootstrap revenue differences. Add A/A simulations before launch and pre-register alpha/stopping. These are experiment design utilities, not evidence of historical Olist treatment effects.

## Decision economics
decision_roi converts an assumed/estimated absolute repeat-rate effect into incremental orders, revenue and contribution using explicit BRL value, margin, customer count and logistics cost. Only supply an identified causal effect or randomized estimate; raw observational differences are not causal ROI.

## Tests
Run pytest -q. The new functions are database-independent. Model fitting and SQL outputs require the configured Olist PostgreSQL source.
