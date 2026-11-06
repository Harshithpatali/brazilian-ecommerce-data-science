from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.io import query_to_dataframe
from src.database.connection import get_engine


REQUIRED_SOURCE_COLUMNS = {
    "olist_orders": {
        "order_id",
        "customer_id",
        "order_status",
        "order_purchase_timestamp",
        "order_approved_at",
        "order_delivered_carrier_date",
        "order_delivered_customer_date",
        "order_estimated_delivery_date",
    },
    "olist_customers": {
        "customer_id",
        "customer_unique_id",
        "customer_city",
        "customer_state",
        "customer_zip_code_prefix",
    },
    "olist_order_items": {
        "order_id",
        "order_item_id",
        "product_id",
        "seller_id",
        "price",
        "freight_value",
    },
    "olist_order_payments": {
        "order_id",
        "payment_value",
        "payment_type",
        "payment_installments",
    },
    "olist_order_reviews": {
        "order_id",
        "review_score",
        "review_answer_timestamp",
    },
    "olist_products": {
        "product_id",
        "product_category_name",
    },
    "olist_sellers": {
        "seller_id",
        "seller_city",
        "seller_state",
    },
    "olist_geolocation": {
        "geolocation_zip_code_prefix",
        "geolocation_lat",
        "geolocation_lng",
        "geolocation_city",
        "geolocation_state",
    },
    "product_category_name_translation": {
        "product_category_name",
        "product_category_name_english",
    },
}


def find_missing_source_columns(
    available_columns: set[tuple[str, str]],
) -> dict[str, list[str]]:
    """Return missing required columns grouped by raw source table."""
    missing: dict[str, list[str]] = {}
    for table, required in REQUIRED_SOURCE_COLUMNS.items():
        absent = sorted(required - {
            column for table_name, column in available_columns if table_name == table
        })
        if absent:
            missing[table] = absent
    return missing


def main() -> None:
    database_url = get_engine().url
    print(
        "Database target (host/database only; credentials redacted): "
        f"{database_url.host}/{database_url.database}"
    )
    source_counts = query_to_dataframe("""
        SELECT
            (SELECT COUNT(*) FROM public.olist_orders) AS source_orders,
            (SELECT COUNT(DISTINCT customer_unique_id) FROM public.olist_customers)
                AS source_customers
    """)
    print("Olist source population:")
    print(source_counts.to_string(index=False))
    if int(source_counts.iloc[0]["source_orders"]) < 90_000:
        raise SystemExit("Source database does not contain the full Olist orders extract.")

    source_columns = query_to_dataframe("""
        SELECT table_name, column_name
        FROM information_schema.columns
        WHERE table_schema = 'public'
          AND table_name IN (
              'olist_orders',
              'olist_customers',
              'olist_order_items',
              'olist_order_payments',
              'olist_order_reviews',
              'olist_products',
              'olist_sellers',
              'olist_geolocation',
              'product_category_name_translation'
          )
    """)
    available = set(zip(source_columns["table_name"], source_columns["column_name"]))
    missing = find_missing_source_columns(available)
    if missing:
        details = "\n".join(
            f"- {table}: {', '.join(columns)}"
            for table, columns in missing.items()
        )
        raise SystemExit(
            "The connected database is not ready for the Olist build. "
            "No project SQL was executed. Missing source columns:\n" + details
        )

    print("Source schema preflight passed.")
    print(f"Verified {len(REQUIRED_SOURCE_COLUMNS)} Olist source tables and required columns.")
    print("No database objects were changed by this preflight.")


if __name__ == "__main__":
    main()
