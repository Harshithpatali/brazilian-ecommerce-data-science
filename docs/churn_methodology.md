# Churn Modeling Methodology

## Prediction question

At the beginning of a monthly snapshot, among customers with at least two
previously delivered orders and delivered activity in the prior 180 days, will
the customer place no order that is eventually delivered during the next 90 days?

This is a forward-looking 90-day inactivity label, not permanent customer loss.

## Snapshot and eligible population

- Customers enter after their second delivered order has reached the customer.
- The first snapshot is the first day of the following calendar month.
- A snapshot is retained only when the customer has at least one order both
  purchased and delivered during the exact prior 180-day feature window.
- Monthly snapshots continue after earlier inactivity labels, allowing
  customers to reactivate.
- A snapshot is eligible only when the source contains a complete 90-day
  forward observation window.

## Target definition

The SQL target uses the half-open interval
[snapshot_date, snapshot_date + 90 days).

- churned = 1 when no eventually-delivered order has a purchase timestamp in
  this interval.
- churned = 0 when at least one eventually-delivered order was purchased in it.
- A purchase exactly at the 90-day upper boundary is outside the interval.
- Observation end is based on the latest purchase timestamp across all source
  orders, not only delivered orders.

Future records create the label only; they are never used as predictors.

## Point-in-time features

For snapshot date t, historical order features use purchase timestamps in
[t - 180 days, t) and require delivery to have occurred before t. The 90-day
and 30-day order/revenue counts are nested windows within the 180-day window.

Other features include average order value, average delivery delay, active
purchase months, recency, order/revenue shares, and order-value ratios. Review
scores are included only when the review-creation timestamp precedes t.

## Temporal evaluation protocol

Unique snapshot dates are split chronologically into a training period, a separate
validation period, and the latest 20% final test period. A 90-day embargo is
applied at each boundary: a training snapshot is retained only when its full
90-day label window ends on or before the first validation date, and a
validation snapshot is retained only when its label window ends on or before
the first final-test date. Dates in the embargo gaps are deliberately excluded.

Expanding-window CV folds are created from unique snapshot dates, not rows.
All customers from one snapshot date are assigned to the same fold side. Each
fold also purges training snapshots whose 90-day target windows overlap the
validation block.
Customers may occur in earlier and later periods; this measures future
predictions for known customers, matching the intended use case.

The candidate model is selected by mean CV average precision (PR-AUC).
The threshold is selected by maximizing F1 on the separate validation period.
Since no business cost matrix is supplied, this is a technical proxy rather
than a proven profit-maximizing intervention threshold.

The frozen model and threshold are evaluated once on the final chronological
holdout. A 90-day embargo separates validation labels from final-test snapshots.
The holdout is not used for model or threshold selection.

## Imbalance and baseline reporting

Report class counts and prevalence for each partition, average precision,
ROC-AUC, thresholded precision/recall/F1, confusion matrix, classification
report, and a DummyClassifier prior-probability average-precision baseline.

Interpret average precision relative to positive-class prevalence. A high
PR-AUC alone is not evidence of useful discrimination when positives are common.

## Limitations

- Evaluation allows customers in earlier training periods to appear in later
  periods. It is not unseen-customer generalization.
- Final delivery status is used retrospectively to define delivered events;
  predictors are restricted to information observable at snapshot time.
- F1-based threshold selection does not encode intervention economics.
- Olist is a historical dataset, not a live marketplace feed.
