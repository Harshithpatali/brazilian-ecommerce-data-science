from sqlalchemy.engine import make_url

from src.database.connection import _build_database_url, _normalize_database_url


def clear_database_env(monkeypatch):
    for name in (
        "DATABASE_URL",
        "SUPABASE_DB_HOST",
        "SUPABASE_DB_PORT",
        "SUPABASE_DB_NAME",
        "SUPABASE_DB_USER",
        "SUPABASE_DB_PASSWORD",
    ):
        monkeypatch.delenv(name, raising=False)


def test_database_url_uses_psycopg_and_supabase_ssl(monkeypatch):
    clear_database_env(monkeypatch)
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql://postgres:secret@db.example.supabase.co:5432/postgres",
    )
    url = _build_database_url()
    assert url.drivername == "postgresql+psycopg"
    assert url.query["sslmode"] == "require"
    assert url.password == "secret"


def test_database_url_preserves_explicit_ssl_mode():
    url = _normalize_database_url(
        "postgresql://postgres:secret@db.example.supabase.co:5432/postgres?sslmode=verify-full"
    )
    assert url.drivername == "postgresql+psycopg"
    assert url.query["sslmode"] == "verify-full"


def test_component_credentials_are_safely_encoded(monkeypatch):
    clear_database_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_DB_HOST", "db.example.supabase.co")
    monkeypatch.setenv("SUPABASE_DB_USER", "postgres")
    monkeypatch.setenv("SUPABASE_DB_PASSWORD", "p@ss:/word#1")
    url = _build_database_url()
    assert url.password == "p@ss:/word#1"
    assert url.query["sslmode"] == "require"
    rendered = url.render_as_string(hide_password=False)
    assert "p%40ss%3A%2Fword%231" in rendered


def test_invalid_port_is_reported(monkeypatch):
    clear_database_env(monkeypatch)
    monkeypatch.setenv("SUPABASE_DB_HOST", "db.example.supabase.co")
    monkeypatch.setenv("SUPABASE_DB_USER", "postgres")
    monkeypatch.setenv("SUPABASE_DB_PASSWORD", "secret")
    monkeypatch.setenv("SUPABASE_DB_PORT", "not-a-port")
    try:
        _build_database_url()
    except RuntimeError as exc:
        assert "SUPABASE_DB_PORT" in str(exc)
    else:
        raise AssertionError("Expected invalid port to raise RuntimeError")
