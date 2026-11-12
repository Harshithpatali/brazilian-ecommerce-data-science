# Decision Brief — Second-Purchase and Customer Retention Intelligence

## Executive decision

**Decision owner:** CRM / customer-retention lead  
**Decision:** Use the platform's SQL customer, cohort and RFM analyses to define and inspect retention cohorts. Do not use the current v6 propensity model to allocate a live campaign: it failed the pre-defined chronological holdout promotion gate.

## Business question

Among customers who have completed a first delivered order, which customers may make a second delivered purchase within the next 90 days, and how could a retention team prioritize a limited contact capacity?

This is a repeat-purchase propensity problem. It is not subscription churn and it is not causal uplift.

## Evidence available

- The analytics platform contains 16 build stages, 20+ analytical tables/views and 7 dashboard pages (see the repository's architecture and dashboard documentation).
- RFM segments, cohort retention and Kaplan–Meier time-to-second-purchase analysis provide descriptive views of customer value and repeat-purchase timing.
- The v6 point-in-time model uses 57 features and purged chronological evaluation.
- On the untouched chronological holdout, v6 PR-AUC was 0.01048 versus 0.01665 for lifetime frequency. The candidate also failed the 5%, 10% and 20% contact-capacity comparisons, so the pipeline correctly marks it research-only.

## Recommended operational use

1. Use cohort and RFM summaries to describe where repeat-purchase activity is concentrated.
2. Use the survival curve to understand the timing of observed second purchases, accounting for right-censoring.
3. Treat campaign scenario outputs as sensitivity analysis only; lift, margin and contact cost are user-entered assumptions.
4. Keep lifetime-frequency as the benchmark until a future model beats it on an untouched later-period holdout and predefined campaign-capacity metrics.

## Stakeholder handoff

A CRM/product review should agree on the eligible customer definition, contact capacity (5%, 10% or 20%), campaign cost, contribution margin, suppression rules and the business outcome to monitor. The dashboard can support that discussion, but historical Olist data does not contain randomized campaign exposure and cannot estimate incremental campaign lift.

## Collaboration and provenance

This is an independently developed portfolio project on public historical marketplace data. The CRM/product role above is the intended decision audience; it is **not a claim of client work, production deployment, or external stakeholder sign-off**.

**Source of truth:** [Interview guide](INTERVIEW_GUIDE.md), [architecture](ARCHITECTURE.md), and [model card](model_card.md).