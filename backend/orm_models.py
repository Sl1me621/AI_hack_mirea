from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Index,
)
from sqlalchemy.orm import relationship

from .db import Base


class Worker(Base):
    __tablename__ = "workers"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=True)

    events = relationship("Event", back_populates="worker")


class Zone(Base):
    __tablename__ = "zones"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False)
    risk_level = Column(Integer, nullable=False)

    events = relationship("Event", back_populates="zone")


class ClothingType(Base):
    __tablename__ = "clothing_types"

    id = Column(Integer, primary_key=True, index=True)
    type_name = Column(String, unique=True, nullable=False)
    color = Column(String, nullable=True)
    description = Column(String, nullable=True)

    events = relationship("Event", back_populates="clothing_type")


class Event(Base):
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, index=True, nullable=False, default=datetime.utcnow)
    event_type = Column(String, index=True, nullable=False)
    worker_id = Column(Integer, ForeignKey("workers.id"), index=True, nullable=True)
    zone_id = Column(Integer, ForeignKey("zones.id"), index=True, nullable=True)
    clothing_type_id = Column(Integer, ForeignKey("clothing_types.id"), index=True, nullable=True)
    train_number = Column(String, nullable=True)
    activity = Column(String, nullable=True)
    is_violation = Column(Boolean, nullable=True)

    worker = relationship("Worker", back_populates="events")
    zone = relationship("Zone", back_populates="events")
    clothing_type = relationship("ClothingType", back_populates="events")


# Composite indexes to support frequent queries
Index("idx_event_worker_timestamp", Event.worker_id, Event.timestamp)
Index("idx_event_zone_timestamp", Event.zone_id, Event.timestamp)
Index("idx_event_activity_timestamp", Event.activity, Event.timestamp)


class WindowAnalytics(Base):
    __tablename__ = "window_analytics"

    id = Column(Integer, primary_key=True, index=True)
    video_id = Column(String, index=True)
    window_start_sec = Column(Float, index=True)
    window_end_sec = Column(Float, index=True)
    workers_in_safe = Column(Integer, nullable=False)
    workers_in_medium = Column(Integer, nullable=False)
    workers_in_danger = Column(Integer, nullable=False)
    dangerous_situations_count = Column(Integer, nullable=False)
    violations_count = Column(Integer, nullable=False)
    cleaning_workers_count = Column(Integer, nullable=False)
    train_present = Column(Boolean, nullable=False)
    train_arrived_in_window = Column(Boolean, nullable=False)
    train_departed_in_window = Column(Boolean, nullable=False)
    train_number = Column(String, nullable=True)
