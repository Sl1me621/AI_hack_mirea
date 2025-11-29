from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import WindowAnalyticsCreate
from .orm_models import WindowAnalytics


def create_window_analytics(db: Session, window: WindowAnalyticsCreate) -> WindowAnalytics:
    obj = WindowAnalytics(**window.model_dump())
    db.add(obj)
    db.commit()
    db.refresh(obj)
    return obj


def create_windows_batch(db: Session, windows: List[WindowAnalyticsCreate]) -> List[WindowAnalytics]:
    created = []
    for window in windows:
        created.append(create_window_analytics(db, window))
    return created


def get_windows(
    db: Session, video_id: Optional[str] = None, limit: int = 100, offset: int = 0
) -> List[WindowAnalytics]:
    query = select(WindowAnalytics).order_by(WindowAnalytics.window_start_sec.asc())
    if video_id:
        query = query.filter(WindowAnalytics.video_id == video_id)
    query = query.limit(limit).offset(offset)
    return list(db.scalars(query).all())


def get_last_window(db: Session, video_id: Optional[str] = None) -> Optional[WindowAnalytics]:
    query = select(WindowAnalytics)
    if video_id:
        query = query.filter(WindowAnalytics.video_id == video_id)
    query = query.order_by(WindowAnalytics.window_end_sec.desc(), WindowAnalytics.id.desc()).limit(1)
    return db.scalars(query).first()
