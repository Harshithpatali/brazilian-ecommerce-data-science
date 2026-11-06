# Model Card — Forward-Looking Customer Churn

## Status

The corrected pipeline version is churn-pit-v2. Existing checked-in model
artifacts were generated before the corrected target and evaluation protocol.
They must be regenerated from the configured PostgreSQL database before the
dashboard can serve the updated churn model.

## Intended use

Portfolio demonstration of point-in-time customer analytics and 90-day
delivered-purchase inactivity risk for customers with at least two previous
delivered orders and delivered activity in the prior 180 days.

This is not a production decision system and should not automatically deny
service or make consequential decisions about customers.

## Target

churned = 1 when no eventually-delivered order has a purchase timestamp in
the half-open interval from the snapshot date up to, but not including, 90
days after the snapshot. A purchase exactly on the 90-day boundary is outside.

## Features

Historical order activity is restricted to the 180 days before the snapshot.
Orders must already have been delivered by the snapshot. Review scores are
included only when their review-creation timestamps precede the snapshot.
Future observations are used only to construct the target.

## Models and evaluation

Candidate models are Logistic Regression and HistGradientBoosting. Model family
and hyperparameters are selected with expanding-window temporal CV grouped by
unique snapshot dates. The threshold is selected on a separate validation
period by maximizing F1. The final chronological holdout is evaluated once.

Reports include class prevalence, average precision, ROC-AUC, thresholded
precision/recall/F1, confusion matrix, and prior-probability dummy baseline.

## Limitations

- The evaluation allows customers in earlier training periods and later
  evaluation periods. It measures future prediction for known customers,
  not unseen-customer generalization.
- Eventual delivery status is used retrospectively to define delivered events.
- F1-based threshold selection does not encode monetary intervention costs.
- The Olist dataset is historical and does not represent a live feed.
