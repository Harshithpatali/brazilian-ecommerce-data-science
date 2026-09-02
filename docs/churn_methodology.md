# Churn Modeling Methodology

## Business question

Among customers who have already demonstrated engagement, which customers are at risk of making no delivered purchase during the next 90 days?

## Modeling population

A customer-month becomes eligible when:
- the customer has at least two delivered orders through that month; and
- the customer has recent delivered activity in the preceding rolling 180-day feature window.

## Snapshot

A monthly activity row is converted into a prediction snapshot on the first day of the following month.

## Features

All predictors are derived from delivered activity on or before the snapshot month:
- 30/90/180-day order counts
- 30/90/180-day revenue
- average order value
- average review score
- average delivery delay
- active months
- recency
- recent order share
- recent revenue share
- orders per active month
- revenue per order
- recent-vs-older order ratio

## Target

`churned = 1` when the customer's next delivered activity is at least 90 days after the snapshot, or there is no later activity within the available data.

`churned = 0` when another delivered activity occurs within the next 90 days.

The last 90 days of the source history are excluded from snapshots so every label has a complete future window.

## Leakage prevention

Future orders are used only to construct the label. Feature calculations are restricted to activity through the snapshot month.

## Evaluation

Use chronological holdout dates. Never randomly mix future snapshots into training.

## Performance design

The expensive raw order history is first materialized into `ml.delivered_orders`, then aggregated to `ml.customer_monthly_activity`. The final churn dataset is built from those compact tables.
