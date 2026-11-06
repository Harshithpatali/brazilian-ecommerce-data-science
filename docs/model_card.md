# Model Card — Forward-Looking Customer Churn

## Status

The corrected pipeline version is churn-pit-v4. The refreshed dataset,
model, metadata, candidate comparison, holdout predictions, classification
report, confusion matrix and precision-recall curve are committed to the
v4 feature branch. Merge PR #4 before deploying this version to the default
branch or the Streamlit application.

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
Identical purchase timestamps are represented by a zero-gap share; positive
gap summaries remain missing when no positive interpurchase interval exists.
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

### Observed v4 holdout performance

The selected HistGradientBoosting model achieved repeat-purchase PR-AUC 0.0472
and ROC-AUC 0.6555 on the final chronological test set. The prior-prevalence
dummy PR-AUC was 0.0242. At the validation-selected F2 threshold, repeat-
purchase precision was 3.62%, recall 65.18%, F1 0.0686 and F2 0.1482.
Balanced accuracy was 61.05% and Matthews correlation was 0.0685.

The test set contained 112 repeat-purchase snapshots out of 4,620 (2.42%).
The confusion matrix was TP=73, FN=39, FP=1,942 and TN=2,566. Thus 43.6% of
test rows were flagged, and most flagged rows were false positives. PR-AUC
was above prevalence but remains low in absolute terms.

Relative to the v3 holdout, recall improved from 7.14% to 65.18% and PR-AUC
from 0.0377 to 0.0472, while precision fell from 5.84% to 3.62%. These results
show a meaningful recall/precision trade-off, not a production-ready model.

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
  intervention costs or false-positive costs. A production decision needs
  a validation-selected precision floor or a known contact-capacity limit.
- The Olist dataset is historical and does not represent a live feed.
