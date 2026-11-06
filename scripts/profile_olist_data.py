from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.io import query_to_dataframe


OUTPUT = ROOT / "reports" / "data"


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # Read-only source inventory: table grain, volume, dates and key completeness.
    inventory = query_to_dataframe("""
        SELECT 'orders' AS table_name, COUNT(*) AS row_count,
               COUNT(DISTINCT order_id) AS distinct_keys,
               COUNT(*) FILTER (WHERE order_id IS NULL) AS null_keys
        FROM public.olist_orders
        UNION ALL
        SELECT 'customers', COUNT(*), COUNT(DISTINCT customer_id),
               COUNT(*) FILTER (WHERE customer_id IS NULL)
        FROM public.olist_customers
        UNION ALL
        SELECT 'order_items', COUNT(*), COUNT(DISTINCT (order_id, order_item_id)),
               COUNT(*) FILTER (WHERE order_id IS NULL OR order_item_id IS NULL)
        FROM public.olist_order_items
        UNION ALL
        SELECT 'payments', COUNT(*), COUNT(DISTINCT (order_id, payment_sequential)),
               COUNT(*) FILTER (WHERE order_id IS NULL)
        FROM public.olist_order_payments
        UNION ALL
        SELECT 'reviews', COUNT(*), COUNT(DISTINCT review_id),
               COUNT(*) FILTER (WHERE review_id IS NULL)
        FROM public.olist_order_reviews
        UNION ALL
        SELECT 'products', COUNT(*), COUNT(DISTINCT product_id),
               COUNT(*) FILTER (WHERE product_id IS NULL)
        FROM public.olist_products
        UNION ALL
        SELECT 'sellers', COUNT(*), COUNT(DISTINCT seller_id),
               COUNT(*) FILTER (WHERE seller_id IS NULL)
        FROM public.olist_sellers
        UNION ALL
        SELECT 'geolocation', COUNT(*), COUNT(*),
               COUNT(*) FILTER (WHERE geolocation_zip_code_prefix IS NULL)
        FROM public.olist_geolocation
        UNION ALL
        SELECT 'category_translation', COUNT(*),
               COUNT(DISTINCT product_category_name),
               COUNT(*) FILTER (WHERE product_category_name IS NULL)
        FROM public.product_category_name_translation
        ORDER BY table_name
    """)

    order_quality = query_to_dataframe("""
        SELECT
          COUNT(*) AS orders,
          COUNT(DISTINCT order_id) AS distinct_orders,
          COUNT(DISTINCT customer_id) AS order_customer_ids,
          COUNT(*) FILTER (WHERE order_status = 'delivered') AS delivered_orders,
          COUNT(*) FILTER (WHERE order_status <> 'delivered') AS non_delivered_orders,
          COUNT(*) FILTER (WHERE order_purchase_timestamp IS NULL) AS missing_purchase_time,
          COUNT(*) FILTER (WHERE order_delivered_customer_date IS NULL) AS missing_delivery_time,
          COUNT(*) FILTER (WHERE order_status = 'delivered'
                            AND order_delivered_customer_date IS NULL) AS delivered_without_delivery_time,
          MIN(order_purchase_timestamp) AS first_purchase,
          MAX(order_purchase_timestamp) AS last_purchase,
          COUNT(*) FILTER (WHERE order_delivered_customer_date < order_purchase_timestamp) AS delivery_before_purchase
        FROM public.olist_orders
    """)

    customer_quality = query_to_dataframe("""
        SELECT
          COUNT(*) AS customer_order_records,
          COUNT(DISTINCT customer_id) AS distinct_customer_ids,
          COUNT(DISTINCT customer_unique_id) AS distinct_real_customers,
          COUNT(*) FILTER (WHERE customer_unique_id IS NULL) AS missing_unique_ids,
          COUNT(DISTINCT customer_city) AS distinct_cities,
          COUNT(DISTINCT customer_state) AS distinct_states
        FROM public.olist_customers
    """)

    repeat_profile = query_to_dataframe("""
        WITH delivered AS (
          SELECT c.customer_unique_id, o.order_id, o.order_purchase_timestamp
          FROM public.olist_orders o
          JOIN public.olist_customers c USING (customer_id)
          WHERE o.order_status = 'delivered'
            AND o.order_delivered_customer_date IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND c.customer_unique_id IS NOT NULL
        ), per_customer AS (
          SELECT customer_unique_id, COUNT(*) AS delivered_orders,
                 MIN(order_purchase_timestamp) AS first_purchase,
                 MAX(order_purchase_timestamp) AS last_purchase
          FROM delivered GROUP BY customer_unique_id
        )
        SELECT
          COUNT(*) AS customers_with_delivered_orders,
          COUNT(*) FILTER (WHERE delivered_orders = 1) AS one_order_customers,
          COUNT(*) FILTER (WHERE delivered_orders >= 2) AS repeat_customers,
          ROUND(100.0 * COUNT(*) FILTER (WHERE delivered_orders >= 2)
                / NULLIF(COUNT(*), 0), 4) AS lifetime_repeat_customer_pct,
          ROUND(AVG(delivered_orders)::numeric, 4) AS avg_delivered_orders,
          percentile_cont(0.5) WITHIN GROUP (ORDER BY delivered_orders) AS median_delivered_orders,
          MAX(delivered_orders) AS max_delivered_orders,
          MIN(first_purchase) AS first_purchase,
          MAX(last_purchase) AS last_purchase
        FROM per_customer
    """)

    repeat_monthly = query_to_dataframe("""
        WITH delivered AS (
          SELECT c.customer_unique_id, o.order_id, o.order_purchase_timestamp,
                 o.order_delivered_customer_date
          FROM public.olist_orders o
          JOIN public.olist_customers c USING (customer_id)
          WHERE o.order_status = 'delivered'
            AND o.order_delivered_customer_date IS NOT NULL
            AND o.order_purchase_timestamp IS NOT NULL
            AND c.customer_unique_id IS NOT NULL
        ), first_order AS (
          SELECT customer_unique_id,
                 MIN(order_purchase_timestamp) AS first_purchase
          FROM delivered GROUP BY customer_unique_id
        ), events AS (
          SELECT f.customer_unique_id,
                 DATE_TRUNC('month', f.first_purchase)::date AS first_purchase_month,
                 COUNT(d.order_id) FILTER (
                   WHERE d.order_purchase_timestamp > f.first_purchase
                 ) AS later_delivered_orders
          FROM first_order f
          LEFT JOIN delivered d USING (customer_unique_id)
          GROUP BY f.customer_unique_id, DATE_TRUNC('month', f.first_purchase)
        )
        SELECT first_purchase_month,
               COUNT(*) AS first_purchase_customers,
               COUNT(*) FILTER (WHERE later_delivered_orders > 0) AS ever_repeat_customers,
               ROUND(100.0 * COUNT(*) FILTER (WHERE later_delivered_orders > 0)
                     / NULLIF(COUNT(*), 0), 4) AS eventual_repeat_pct
        FROM events
        GROUP BY first_purchase_month
        ORDER BY first_purchase_month
    """)

    source_distributions = {
        "order_status": query_to_dataframe("""
            SELECT order_status, COUNT(*) AS rows
            FROM public.olist_orders GROUP BY order_status ORDER BY rows DESC
        """),
        "customer_state": query_to_dataframe("""
            SELECT customer_state AS state, COUNT(DISTINCT customer_unique_id) AS customers
            FROM public.olist_customers
            GROUP BY customer_state ORDER BY customers DESC
        """),
        "payment_type": query_to_dataframe("""
            SELECT payment_type, COUNT(*) AS payment_rows,
                   COUNT(DISTINCT order_id) AS orders,
                   ROUND(AVG(payment_value)::numeric, 2) AS avg_payment_value
            FROM public.olist_order_payments
            GROUP BY payment_type ORDER BY payment_rows DESC
        """),
        "review_score": query_to_dataframe("""
            SELECT review_score, COUNT(*) AS review_rows
            FROM public.olist_order_reviews
            GROUP BY review_score ORDER BY review_score
        """),
        "top_product_categories": query_to_dataframe("""
            SELECT COALESCE(p.product_category_name, 'unknown') AS category,
                   COUNT(*) AS item_rows,
                   COUNT(DISTINCT oi.order_id) AS orders
            FROM public.olist_order_items oi
            LEFT JOIN public.olist_products p USING (product_id)
            GROUP BY 1 ORDER BY item_rows DESC LIMIT 30
        """),
    }

    payload = {
        "profile_scope": "read-only raw Olist source tables",
        "source_inventory": inventory.to_dict(orient="records"),
        "order_quality": order_quality.to_dict(orient="records")[0],
        "customer_quality": customer_quality.to_dict(orient="records")[0],
        "lifetime_repeat_purchase": repeat_profile.to_dict(orient="records")[0],
        "interpretation": {
            "customer_key": "customer_unique_id",
            "delivered_order_definition": "order_status=delivered and delivery timestamp is not null",
            "warning": "Lifetime repeat rate is not the same as second purchase within 90 days.",
        },
        "distributions": {
            name: frame.to_dict(orient="records")
            for name, frame in source_distributions.items()
        },
    }
    (OUTPUT / "olist_source_profile.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )
    repeat_monthly.to_csv(OUTPUT / "repeat_purchase_by_first_order_month.csv", index=False)
    inventory.to_csv(OUTPUT / "raw_table_inventory.csv", index=False)

    print(json.dumps({
        "source_inventory": payload["source_inventory"],
        "order_quality": payload["order_quality"],
        "customer_quality": payload["customer_quality"],
        "lifetime_repeat_purchase": payload["lifetime_repeat_purchase"],
        "profile_file": str(OUTPUT / "olist_source_profile.json"),
        "monthly_repeat_file": str(OUTPUT / "repeat_purchase_by_first_order_month.csv"),
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
