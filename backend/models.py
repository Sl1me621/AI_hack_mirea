from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from pydantic import BaseModel, ConfigDict


class EventCreate(BaseModel):
    timestamp: datetime
    event_type: str
    worker_id: Optional[int] = None
    zone_name: Optional[str] = None
    clothing_type_name: Optional[str] = None
    train_number: Optional[str] = None
    activity: Optional[str] = None
    is_violation: Optional[bool] = None


class EventRead(BaseModel):
    id: int
    timestamp: datetime
    event_type: str
    worker_id: Optional[int] = None
    zone_name: Optional[str] = None
    clothing_type_name: Optional[str] = None
    train_number: Optional[str] = None
    activity: Optional[str] = None
    is_violation: Optional[bool] = None

    model_config = ConfigDict(from_attributes=True)


class CurrentState(BaseModel):
    workers_in_safe: int
    workers_in_medium: int
    workers_in_danger: int
    dangerous_situations_count: int
    cleaning_workers_count: int


class ZoneRead(BaseModel):
    id: int
    name: str
    risk_level: int

    model_config = ConfigDict(from_attributes=True)


class ClothingTypeRead(BaseModel):
    id: int
    type_name: str
    color: Optional[str] = None
    description: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class WindowAnalyticsBase(BaseModel):
    video_id: str
    window_start_sec: float
    window_end_sec: float
    workers_in_safe: int
    workers_in_medium: int
    workers_in_danger: int
    dangerous_situations_count: int
    violations_count: int
    cleaning_workers_count: int
    train_present: bool
    train_arrived_in_window: bool
    train_departed_in_window: bool
    train_number: Optional[str] = None


class WindowAnalyticsCreate(WindowAnalyticsBase):
    pass


class WindowAnalyticsRead(WindowAnalyticsBase):
    id: int

    model_config = ConfigDict(from_attributes=True)


class WindowAnalyticsBatchCreate(BaseModel):
    windows: List[WindowAnalyticsCreate]
