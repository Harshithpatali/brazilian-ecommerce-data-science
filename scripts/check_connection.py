import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.database.connection import get_engine
from sqlalchemy import text

with get_engine().connect() as conn:
    result = conn.execute(text("SELECT current_database(), current_user, now()"))
    db, user, now = result.one()

print(f"Connected successfully")
print(f"Database: {db}")
print(f"User: {user}")
print(f"Server time: {now}")
