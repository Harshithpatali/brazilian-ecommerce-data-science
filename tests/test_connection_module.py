from src.database.connection import get_engine

def test_connection_factory_exists():
    assert callable(get_engine)
