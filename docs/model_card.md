# Model Card — Forward-Looking Customer Churn

## Status

The corrected pipeline version is churn-pit-v4. Existing v3 artifacts are
superseded by the new point-in-time feature set and recall-oriented threshold
policy. The v4 model and metadata must be regenerated before deployment.

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

Short-term order activity uses the 180 days before the snapshot, while a
separate prior 180-day window supplies purchase-frequency and revenue trends.
Lifetime delivered-order count, customer tenure and interpurchase-gap
statistics use only orders purchased and delivered before the snapshot.
Review scores are included only when their review-answer timestamps precede
the snapshot. Future observations are used only to construct the target.

## Class balance and evaluation

The rebuilt dataset contains 306 repeat-purchase snapshots out of 10,570
(2.89%); 97.11% have no delivered purchase in the next 90 days. Model family
and hyperparameters are selected with expanding-window temporal CV average
precision for the minority repeat-purchase event (churned = 0).

The candidate families are Logistic Regression, HistGradientBoosting and
Random Forest. The estimator outputs churn probability; repeat-purchase
probability is its complement. The repeat-purchase threshold is selected on
a separate validation period by maximum F2 (beta=2, recall-weighted) and
converted to the complementary churn-probability cutoff for the dashboard.

Reports include repeat-purchase prevalence, PR-AUC, ROC-AUC, both classes'
precision/recall/F1, F2, balanced accuracy, Matthews correlation coefficient,
confusion matrix, and a prior-probability dummy baseline. PR-AUC must be
compared with repeat-purchase prevalence.

## Limitations

- Evaluation allows customers in earlier training periods and later
  evaluation periods. It measures future prediction for known customers,
  not unseen-customer generalization.
- Eventual delivery status is used retrospectively to define delivered events.
- F2 threshold selection favors recall but does not encode monetary
  intervention costs or false-positive costs.
- The Olist dataset is historical and does not represent a live feed.
