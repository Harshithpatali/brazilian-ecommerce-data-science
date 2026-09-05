from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FOLDERS = [
    "sql/03_data_model",
    "sql/04_kpi_layer",
    "sql/05_customer_analytics",
    "sql/06_cohort_retention",
    "sql/07_product_analytics",
    "sql/08_seller_analytics",
    "sql/09_delivery_analytics",
    "sql/10_customer_experience",
    "sql/11_geographic_analytics",
    "sql/12_advanced_analytics",
    "sql/13_statistical_analysis",
    "sql/14_ml_features",
    "sql/15_reporting",
]

for folder in FOLDERS:
    subprocess.run(
        [sys.executable, "-m", "scripts.run_sql_folder", str(ROOT / folder)],
        check=True,
        cwd=ROOT,
    )

subprocess.run(
    [sys.executable, "-m", "scripts.build_dashboard_data"],
    check=True,
    cwd=ROOT,
)
subprocess.run(
    [sys.executable, "-m", "scripts.build_ml_data"],
    check=True,
    cwd=ROOT,
)
subprocess.run(
    [sys.executable, "-m", "src.statistical_analysis"],
    check=True,
    cwd=ROOT,
)

print("Project build completed.")
