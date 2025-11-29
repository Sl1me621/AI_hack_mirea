from __future__ import annotations

from pathlib import Path
import sys

from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.db import Base, engine, SessionLocal  # noqa: E402
from backend.orm_models import ClothingType, Zone, WindowAnalytics  # noqa: E402

ZONES = [
    ("safe", 1),
    ("medium", 2),
    ("danger", 3),
]

CLOTHING_TYPES = [
    ("worker_white", "white", "Белый комбинезон — бригада ТО"),
    ("worker_orange", "orange", "Оранжевый комбинезон — уборка"),
    ("worker_color3", "blue", "Работник тип 3"),
    ("worker_color4", "green", "Работник тип 4"),
]


def init_base():
    Base.metadata.create_all(bind=engine)


def seed_reference_data():
    session = SessionLocal()
    try:
        existing_zones = {z.name for z in session.scalars(select(Zone)).all()}
        for name, risk in ZONES:
            if name not in existing_zones:
                session.add(Zone(name=name, risk_level=risk))

        existing_types = {ct.type_name for ct in session.scalars(select(ClothingType)).all()}
        for type_name, color, description in CLOTHING_TYPES:
            if type_name not in existing_types:
                session.add(ClothingType(type_name=type_name, color=color, description=description))

        session.commit()
    finally:
        session.close()


if __name__ == "__main__":
    init_base()
    seed_reference_data()
    print("Database initialized and reference data seeded.")
