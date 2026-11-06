from __future__ import annotations

import os
from functools import lru_cache

from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import URL, Engine, make_url

load_dotenv()


def _normalize_database_url(raw_url: str) -> URL:
    """Normalize PostgreSQL URLs for psycopg 3 and require TLS for Supabase."""
    url = make_url(raw_url.strip())
    if url.drivername in {"postgres", "postgresql"}:
        url = url.set(drivername="postgresql+psycopg")

    query = dict(url.query)
    host = (url.host or "").lower()
    is_supabase = host.endswith(".supabase.co") or ".pooler.supabase.com" in host
    if is_supabase:
        query.setdefault("sslmode", "require")
    return url.set(query=query)


def _build_database_url() -> URL:
    database_url = os.getenv("DATABASE_URL", "").strip()
    if database_url:
        try:
            return _normalize_database_url(database_url)
        except Exception as exc:
            raise RuntimeError(
                "DATABASE_URL is not a valid SQLAlchemy PostgreSQL URL. "
                "Use postgresql:// or postgresql+psycopg://."
            ) from exc

    host = os.getenv("SUPABASE_DB_HOST", "").strip()
    port_raw = os.getenv("SUPABASE_DB_PORT", "5432").strip()
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

    try:
        port = int(port_raw)
    except ValueError as exc:
        raise RuntimeError("SUPABASE_DB_PORT must be an integer.") from exc
    if not 1 <= port <= 65535:
        raise RuntimeError("SUPABASE_DB_PORT must be between 1 and 65535.")

    query = {}
    host_lower = host.lower()
    if host_lower.endswith(".supabase.co") or ".pooler.supabase.com" in host_lower:
        query["sslmode"] = "require"

    # URL.create safely encodes passwords containing @, :, /, #, or other
    # reserved URL characters. Do not build credentials with string formatting.
    return URL.create(
        drivername="postgresql+psycopg",
        username=user,
        password=password,
        host=host,
        port=port,
        database=database,
        query=query,
    )


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    try:
        connect_timeout = int(os.getenv("DB_CONNECT_TIMEOUT", "10"))
    except ValueError as exc:
        raise RuntimeError("DB_CONNECT_TIMEOUT must be a positive integer.") from exc
    if connect_timeout <= 0:
        raise RuntimeError("DB_CONNECT_TIMEOUT must be a positive integer.")

    return create_engine(
        _build_database_url(),
        pool_pre_ping=True,
        pool_recycle=1800,
        connect_args={"connect_timeout": connect_timeout},
        future=True,
    )
