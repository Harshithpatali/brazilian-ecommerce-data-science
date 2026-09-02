from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine

load_dotenv()


def _build_database_url() -> str:
    url = os.getenv("DATABASE_URL", "").strip()
    if url:
        return url

    host = os.getenv("SUPABASE_DB_HOST", "").strip()
    port = os.getenv("SUPABASE_DB_PORT", "5432").strip()
    database = os.getenv("SUPABASE_DB_NAME", "postgres").strip()
    user = os.getenv("SUPABASE_DB_USER", "postgres").strip()
    password = os.getenv("SUPABASE_DB_PASSWORD", "").strip()

    missing = [
        name for name, value in {
            "SUPABASE_DB_HOST": host,
            "SUPABASE_DB_NAME": database,
            "SUPABASE_DB_USER": user,
            "SUPABASE_DB_PASSWORD": password,
        }.items() if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing database configuration. Set DATABASE_URL or: "
            + ", ".join(missing)
        )

    return (
        f"postgresql+psycopg://{user}:{password}"
        f"@{host}:{port}/{database}"
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    return create_engine(
        _build_database_url(),
        pool_pre_ping=True,
        future=True,
    )
