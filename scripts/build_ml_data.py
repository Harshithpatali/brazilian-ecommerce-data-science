from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.io import save_query_to_parquet

output = save_query_to_parquet(
    """
    SELECT *
    FROM ml.customer_churn_dataset
    ORDER BY snapshot_date, customer_unique_id
    """,
    ROOT / "data" / "ml" / "customer_churn_dataset.parquet",
)

print(f"Wrote {output}")
