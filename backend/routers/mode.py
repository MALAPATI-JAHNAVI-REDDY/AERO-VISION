
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.services.camera_service import camera_service

router = APIRouter(tags=["Operating Mode"])

active_mode = "live"


class ModeRequest(BaseModel):
    mode: str


@router.get("/mode")
def get_mode():
    return {
        "mode": active_mode,
        "camera_running": camera_service.running
    }


@router.post("/mode")
def change_mode(request: ModeRequest):
    global active_mode

    if request.mode not in {"live", "media"}:
        raise HTTPException(
            status_code=400,
            detail="Mode must be 'live' or 'media'."
        )

    if request.mode == active_mode:
        return get_mode()

    if request.mode == "media":
        camera_service.stop()
        active_mode = "media"

    else:
        try:
            camera_service.start()
            active_mode = "live"
        except Exception as exc:
            active_mode = "media"
            raise HTTPException(
                status_code=500,
                detail=f"Could not start live camera: {exc}"
            )

    print(f"[SYSTEM] Operating mode changed to: {active_mode}")

    return get_mode()
