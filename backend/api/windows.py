from __future__ import annotations

from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from .. import crud
from ..db import get_db
from ..models import WindowAnalyticsBatchCreate, WindowAnalyticsRead

router = APIRouter(prefix="/api/windows", tags=["windows"])


@router.post("/batch", response_model=List[WindowAnalyticsRead])
def create_windows_batch(payload: WindowAnalyticsBatchCreate, db: Session = Depends(get_db)):
    created = crud.create_windows_batch(db, payload.windows)
    return created


@router.get("/", response_model=List[WindowAnalyticsRead])
def list_windows(
    video_id: Optional[str] = Query(default=None),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
):
    return crud.get_windows(db, video_id=video_id, limit=limit, offset=offset)
