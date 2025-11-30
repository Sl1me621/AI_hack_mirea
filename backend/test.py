# Быстрый пинг БД через SQLAlchemy (по умолчанию SQLite в корне проекта)
from pathlib import Path

from sqlalchemy import create_engine, text

db_path = Path(__file__).resolve().parents[1] / "depot_analytics.db"
url = f"sqlite:///{db_path}"
print("URL:", repr(url))
engine = create_engine(url, future=True, connect_args={"check_same_thread": False})
with engine.connect() as conn:
    print("SELECT 1 ->", conn.execute(text("SELECT 1")).scalar())
