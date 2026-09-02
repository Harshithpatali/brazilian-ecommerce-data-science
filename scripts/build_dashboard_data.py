from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pathlib import Path

from src.database.io import save_query_to_csv

OUT = ROOT / "data" / "dashboard"

DATASETS = {
    "executive_kpis.csv": "SELECT * FROM analytics.executive_kpis",
    "monthly_revenue.csv": "SELECT * FROM analytics.monthly_revenue ORDER BY month",
    "customer_metrics.csv": """
        SELECT * FROM analytics.customer_metrics
        ORDER BY revenue DESC NULLS LAST
    """,
    "customer_segments.csv": """
        SELECT * FROM analytics.customer_segments
        ORDER BY revenue DESC
    """,
    "cohort_retention.csv": """
        SELECT * FROM analytics.cohort_retention
        ORDER BY cohort_month, months_since_first_order
    """,
    "category_performance.csv": """
        SELECT * FROM analytics.category_performance
        ORDER BY revenue DESC
    """,
    "seller_performance.csv": """
        SELECT * FROM analytics.seller_performance
        ORDER BY revenue DESC
    """,
    "delivery_performance.csv": """
        SELECT * FROM analytics.delivery_performance
        ORDER BY month
    """,
    "delivery_vs_review.csv": """
        SELECT * FROM analytics.delivery_vs_review
    """,
    "geographic_performance.csv": """
        SELECT * FROM analytics.geographic_performance
        ORDER BY revenue DESC
    """,
}

for filename, sql in DATASETS.items():
    output = save_query_to_csv(sql, OUT / filename)
    print(f"Wrote {output}")
