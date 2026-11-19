
from fastapi import APIRouter, Query

from backend.services.event_service import (
    read_recent_events,
    get_event_count
)

router = APIRouter(prefix="/detections", tags=["Detections"])


@router.get("/")
def recent_detections(
    limit: int = Query(default=50, ge=1, le=500)
):
    return {
        "total_events": get_event_count(),
        "events": read_recent_events(limit)
    }
