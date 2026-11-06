# Second-Purchase Propensity Methodology

## Prediction question

At the beginning of a monthly customer snapshot, after the customer's first
delivered order, will the customer place a second order that is eventually
delivered during the following 90 days?

This is a second-purchase propensity target, not permanent customer loss.
The SQL column remains named \`churned\` for compatibility with the project:
\`churned = 0\` is the positive second-purchase event; \`churned = 1\` means no
second delivered purchase is observed in the 90-day window.

## Snapshot eligibility and cohort entry

- Customers enter after their **first delivered order** has reached the customer.
- Their first prediction snapshot is the first day of the following calendar month.
- A snapshot is retained only when at least one order was both purchased and
  delivered during the exact prior 180-day feature window.
- Monthly snapshots continue after earlier no-purchase labels, allowing
  customers to reactivate and later make a second purchase.
- A snapshot is retained only when the complete 90-day future outcome window
  is observable in the source data.
- The unique grain is \`(customer_unique_id, snapshot_date)\`.

## Target definition

The SQL target uses the half-open interval
\`[snapshot_date, snapshot_date + 90 days)\`.

- \`churned = 0\` when at least one eventually-delivered order has a purchase
  timestamp inside this interval. This is the second-purchase event for the
  first-delivered-order cohort.
- \`churned = 1\` when no eventually-delivered order has a purchase timestamp
  inside the interval.
- A purchase exactly at the 90-day upper boundary is outside the interval.
- The observation end is based on the latest purchase timestamp across all
  source orders, not only delivered orders.
- Future observations are used only to construct the label, never as predictors.

## Point-in-time features

For snapshot date \(t\), recent order features use purchase timestamps in
\`[t - 180 days, t)\` and require delivery to have occurred before \(t\).
The 90-day and 30-day order/revenue counts are nested windows inside the
180-day window.

Additional features include:
- prior activity in `[t - 360 days, t - 180 days)`;
- first-order basket value, item/product/category/seller counts and freight share;
- first-order payment type, payment installments and credit-card/boleto/debit/voucher indicators;
- first-order approval lag, delivery duration, delivery delay and review score;
- rolling 180-day averages for basket size, category/seller diversity, installments, credit-card share and freight share;
- lifetime delivered-order count and customer tenure;
- positive interpurchase-gap count, average/median positive gap, gap
  coefficient of variation, zero-gap share for simultaneous purchase
  timestamps, and recency relative to the average positive gap;
- review scores, available only when \`review_answer_timestamp < t\`;
- order value, delivery delay, active months and recent-vs-prior purchase ratios.

Every historical purchase and delivery used by these features is strictly
before the snapshot. Two orders with identical purchase timestamps are
represented by the zero-gap share. Positive-gap statistics remain undefined
when there is no positive gap; the fitted pipeline imputes them rather than
inventing a zero-day purchase cycle.

## Data-population audit

A read-only audit of the connected Olist source found 99,441 source orders,
96,470 delivered orders and 93,350 customers with at least one delivered
order. The first-delivered-order cohort query produced 357,413 eligible
customer-month snapshots across 79,594 customers, with 2,686 second-purchase
events (0.752%).

These are database cohort-audit counts, not final v5 partition or model
metrics. The automated pipeline requires at least 300,000 rows, 70,000
customers and 2,000 positive events before training. This guard is designed
to fail when an old third-or-later-purchase cohort or truncated extract is
accidentally supplied.

## Temporal evaluation protocol

- Snapshot dates are split chronologically into training, validation and
  final test periods.
- A 90-day embargo separates periods so earlier label windows cannot overlap
  a later partition.
- Expanding-window CV splits unique snapshot dates, not individual rows.
  Customers from one date remain on the same side of a fold.
- Each CV fold purges training snapshots whose 90-day label windows overlap
  the validation block.
- Hyperparameters are selected by mean CV average precision (PR-AUC) for the
  second-purchase event (\`1 - churned\`).
- The final chronological test is evaluated once and is not used for model,
  calibration or threshold selection.

This design measures future predictions for both customers already seen in
development and customers first observed in the test period. The latter are
reported separately; they must not be conflated with known-customer
performance.

## Probability calibration and campaign ranking

The estimator is calibrated with a Platt sigmoid fitted on purged
out-of-fold predictions from the training partition only. Validation selects
the diagnostic F2 operating threshold. The final test is used to report
calibration and ranking metrics, never to fit the calibrator or choose a
threshold.

The evaluation reports:
- PR-AUC and ROC-AUC versus prior-probability, recency-only,
  recent-frequency-only and lifetime-frequency-only baselines;
- precision, recall, F1, F2, balanced accuracy and Matthews correlation;
- Brier score, log-loss, expected calibration error and calibration bins;
- monthly top-5%, top-10% and top-20% campaign-capacity precision, recall,
  captured-event count and lift over monthly prevalence;
- a customer-cluster bootstrap 95% PR-AUC interval;
- a separate final-test result for customers absent from both training
  and validation.

The monthly capacity metrics rank customers within each monthly cohort.
They are more directly useful for a campaign with finite contact capacity
than an unrestricted global classification threshold.

## Limitations

- Olist is a historical marketplace dataset, not a live customer feed.
- Eventual delivery is used retrospectively to define delivered-purchase
  events; predictors remain restricted to information observable at snapshot time.
- The positive event is rare. Report PR-AUC relative to the positive
  prevalence and include simple recency/frequency baselines.
- Probability calibration and offline ranking do not prove incremental
  campaign lift. A randomized retention experiment is required to measure
  causal impact and return on investment.
- The F2 threshold is a technical diagnostic, not a business-optimal contact
  policy. Real deployment needs a campaign budget, intervention cost,
  minimum precision target and randomized outcome measurement.
