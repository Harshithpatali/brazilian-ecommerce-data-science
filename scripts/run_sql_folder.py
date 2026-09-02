from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import sys
from pathlib import Path

from sqlalchemy import text

from src.database.connection import get_engine


def run_folder(folder: Path) -> None:
    sql_files = sorted(folder.glob("*.sql"))
    if not sql_files:
        raise SystemExit(f"No SQL files found in {folder}")

    engine = get_engine()
    for path in sql_files:
        sql = path.read_text(encoding="utf-8")
        print(f"Running {path}")
        with engine.begin() as conn:
            conn.execute(text(sql))
        print(f"Done: {path.name}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python scripts/run_sql_folder.py path/to/sql_folder")
    run_folder(Path(sys.argv[1]))
