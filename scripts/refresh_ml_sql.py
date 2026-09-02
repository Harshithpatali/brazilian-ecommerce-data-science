from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

subprocess.run(
    [
        sys.executable,
        "-m",
        "scripts.run_sql_folder",
        str(ROOT / "sql" / "14_ml_features"),
    ],
    check=True,
    cwd=ROOT,
)

print("Optimized ML dataset refreshed in Supabase.")
