from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.io import query_to_dataframe

df = query_to_dataframe("""
SELECT
    COUNT(*) AS rows,
    COUNT(DISTINCT customer_unique_id) AS customers,
    COUNT(DISTINCT snapshot_date) AS snapshots,
    MIN(snapshot_date) AS first_snapshot,
    MAX(snapshot_date) AS last_snapshot,
    AVG(churned::numeric) AS churn_rate
FROM ml.customer_churn_dataset
""")

print(df.to_string(index=False))
