# Data Model

## Raw layer

The raw `public.olist_*` tables remain unchanged.

## Analytical layer

`analytics.order_level` is the central order-grain dataset.

It aggregates:
- order items to one row per order
- payments to one row per order
- reviews to one row per order

This prevents many-to-many multiplication when combining order-level measures.

## Customer identity

For customer-level analysis, use `customer_unique_id`, not only `customer_id`.

`customer_id` represents the customer record attached to an order in the original dataset, while `customer_unique_id` is the stable customer identity used to combine purchase history across orders.

## Model layers

```text
public raw tables
       |
       v
analytics.order_level
       |
       +--> analytics.monthly_revenue
       +--> analytics.customer_metrics
       +--> analytics.customer_rfm
       +--> analytics.cohort_retention
       +--> analytics.category_performance
       +--> analytics.seller_performance
       +--> analytics.delivery_performance
       +--> analytics.delivery_vs_review
       +--> analytics.geographic_performance
       |
       v
ml.customer_features
```
