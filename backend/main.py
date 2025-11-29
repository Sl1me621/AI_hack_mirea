from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import and_, select
from sqlalchemy.orm import Session, selectinload

from . import crud
from .api import windows as windows_router
from .db import get_db
from .models import (
    ClothingTypeRead,
    CurrentState,
    EventCreate,
    EventRead,
    WindowAnalyticsRead,
    ZoneRead,
)
from .orm_models import ClothingType, Event, Worker, Zone, WindowAnalytics

app = FastAPI(title="Depot Monitoring MVP", version="0.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost",
        "http://127.0.0.1",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def event_to_read(model: Event) -> EventRead:
    return EventRead(
        id=model.id,
        timestamp=model.timestamp,
        event_type=model.event_type,
        worker_id=model.worker_id,
        zone_name=model.zone.name if model.zone else None,
        clothing_type_name=model.clothing_type.type_name if model.clothing_type else None,
        train_number=model.train_number,
        activity=model.activity,
        is_violation=model.is_violation,
    )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/events", response_model=EventRead)
def create_event(payload: EventCreate, db: Session = Depends(get_db)):
    zone = None
    clothing_type = None

    if payload.zone_name:
        zone = db.query(Zone).filter(Zone.name == payload.zone_name).first()
        if not zone:
            raise HTTPException(status_code=400, detail="zone_name не найдена")

    if payload.clothing_type_name:
        clothing_type = (
            db.query(ClothingType).filter(ClothingType.type_name == payload.clothing_type_name).first()
        )
        if not clothing_type:
            raise HTTPException(status_code=400, detail="clothing_type_name не найден")

    worker = None
    if payload.worker_id is not None:
        worker = db.query(Worker).filter(Worker.id == payload.worker_id).first()
        if not worker:
            worker = Worker(id=payload.worker_id, name=f"worker_{payload.worker_id}")
            db.add(worker)
            db.flush()

    event = Event(
        timestamp=payload.timestamp,
        event_type=payload.event_type,
        worker=worker,
        zone=zone,
        clothing_type=clothing_type,
        train_number=payload.train_number,
        activity=payload.activity,
        is_violation=payload.is_violation,
    )
    db.add(event)
    db.commit()
    db.refresh(event)

    return event_to_read(event)


@app.get("/api/events", response_model=List[EventRead])
def get_events(
    start_time: Optional[datetime] = Query(default=None),
    end_time: Optional[datetime] = Query(default=None),
    event_type: Optional[str] = Query(default=None),
    zone_name: Optional[str] = Query(default=None),
    worker_id: Optional[int] = Query(default=None),
    activity: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    query = db.query(Event).options(
        selectinload(Event.zone), selectinload(Event.clothing_type)
    )

    filters = []
    if start_time:
        filters.append(Event.timestamp >= start_time)
    if end_time:
        filters.append(Event.timestamp <= end_time)
    if event_type:
        filters.append(Event.event_type == event_type)
    if worker_id is not None:
        filters.append(Event.worker_id == worker_id)
    if activity:
        filters.append(Event.activity == activity)
    if zone_name:
        query = query.join(Zone, isouter=True)
        filters.append(Zone.name == zone_name)

    if filters:
        query = query.filter(and_(*filters))

    events = query.order_by(Event.timestamp.asc()).all()
    return [event_to_read(e) for e in events]


@app.get("/api/metrics/current_state", response_model=CurrentState)
def get_current_state(
    video_id: Optional[str] = Query(default=None), db: Session = Depends(get_db)
):
    last_window = crud.get_last_window(db, video_id=video_id)
    if not last_window:
        return CurrentState(
            workers_in_safe=0,
            workers_in_medium=0,
            workers_in_danger=0,
            dangerous_situations_count=0,
            cleaning_workers_count=0,
        )
    return CurrentState(
        workers_in_safe=last_window.workers_in_safe,
        workers_in_medium=last_window.workers_in_medium,
        workers_in_danger=last_window.workers_in_danger,
        dangerous_situations_count=last_window.dangerous_situations_count,
        cleaning_workers_count=last_window.cleaning_workers_count,
    )


@app.get("/api/metrics/dangerous_situations", response_model=List[WindowAnalyticsRead])
def get_dangerous_situations(
    video_id: Optional[str] = Query(default=None),
    db: Session = Depends(get_db),
):
    query = select(WindowAnalytics).filter(WindowAnalytics.dangerous_situations_count > 0)
    if video_id:
        query = query.filter(WindowAnalytics.video_id == video_id)
    query = query.order_by(WindowAnalytics.window_start_sec.asc())
    windows = list(db.scalars(query).all())
    return windows


@app.get("/api/events/cleaning", response_model=List[EventRead])
def get_cleaning_events(db: Session = Depends(get_db)):
    events = (
        db.query(Event)
        .options(selectinload(Event.zone), selectinload(Event.clothing_type))
        .filter(Event.activity == "cleaning")
        .order_by(Event.timestamp.asc())
        .all()
    )
    return [event_to_read(e) for e in events]


@app.get("/api/zones", response_model=List[ZoneRead])
def list_zones(db: Session = Depends(get_db)):
    zones = db.query(Zone).order_by(Zone.risk_level.asc()).all()
    return zones


@app.get("/api/clothing_types", response_model=List[ClothingTypeRead])
def list_clothing_types(db: Session = Depends(get_db)):
    types = db.query(ClothingType).order_by(ClothingType.id.asc()).all()
    return types


app.include_router(windows_router.router)
