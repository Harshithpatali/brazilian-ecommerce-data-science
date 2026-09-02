from __future__ import annotations

from pathlib import Path

import pandas as pd
from sqlalchemy import text

from .connection import get_engine


def query_to_dataframe(sql: str, params: dict | None = None) -> pd.DataFrame:
    with get_engine().connect() as conn:
        return pd.read_sql(text(sql), conn, params=params or {})


def execute_sql(sql: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(text(sql))


def save_query_to_csv(sql: str, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = query_to_dataframe(sql)
    df.to_csv(output, index=False)
    return output


def save_query_to_parquet(sql: str, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df = query_to_dataframe(sql)
    df.to_parquet(output, index=False)
    return output
