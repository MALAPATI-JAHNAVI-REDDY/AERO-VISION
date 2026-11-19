
import sys
import json
import base64
import threading
import time
import asyncio

from pathlib import Path

import cv2
import numpy as np

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, UploadFile, File, HTTPException

from backend.config import EVENT_LOG

router = APIRouter(tags=["Live"])

# Load the existing Pi-agent inference modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PI_AGENT_DIR = PROJECT_ROOT / "pi-agent"

if str(PI_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(PI_AGENT_DIR))

from inference.detector import DroneVisionDetector
from inference.tracker import IoUTracker
from events.detection_event import create_detection_event

CONFIG_PATH = PI_AGENT_DIR / "config" / "model_config.yaml"

# Shared model and tracker: initialized only once.
detector = None
tracker = IoUTracker(iou_threshold=0.3, max_age=10)
inference_lock = threading.Lock()
frame_counter = 0


def get_detector():
    global detector

    if detector is None:
        detector = DroneVisionDetector(CONFIG_PATH)

    return detector


@router.post("/live/frame")
async def analyze_live_frame(file: UploadFile = File(...)):
    global frame_counter

    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Expected an image frame.")

    contents = await file.read()

    if not contents:
        raise HTTPException(status_code=400, detail="Empty frame received.")

    image_array = np.frombuffer(contents, dtype=np.uint8)
    frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

    if frame is None:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    try:
        with inference_lock:
            model = get_detector()

            start_time = time.perf_counter()

            detections = model.detect(frame)
            tracked = tracker.update(detections)
            annotated = model.annotate(frame, tracked)

            inference_ms = (time.perf_counter() - start_time) * 1000
            frame_counter += 1

            success, encoded = cv2.imencode(
                ".jpg",
                annotated,
                [cv2.IMWRITE_JPEG_QUALITY, 85]
            )

            if not success:
                raise RuntimeError("Could not encode annotated frame.")

            image_base64 = base64.b64encode(
                encoded.tobytes()
            ).decode("utf-8")

            event = create_detection_event(
                detections=tracked,
                frame_id=frame_counter,
                camera_source="browser_webcam"
            )

            event["visible_objects"] = len(tracked)
            event["tracked_objects"] = tracker.get_active_count()
            event["unique_objects"] = len({
                item.get("track_id")
                for item in tracked
                if item.get("track_id") is not None
            })

            EVENT_LOG.parent.mkdir(parents=True, exist_ok=True)

            with EVENT_LOG.open("a", encoding="utf-8") as log_file:
                log_file.write(
                    json.dumps(event, ensure_ascii=False) + "\n"
                )

        return {
            "frame_id": frame_counter,
            "detections": tracked,
            "object_count": len(tracked),
            "inference_ms": round(inference_ms, 2),
            "annotated_image": f"data:image/jpeg;base64,{image_base64}",
            "activity_event": event
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Live frame inference failed: {exc}"
        )


@router.websocket("/ws/detections")
async def detection_stream(websocket: WebSocket):
    await websocket.accept()

    try:
        position = EVENT_LOG.stat().st_size if EVENT_LOG.exists() else 0

        await websocket.send_json({
            "type": "connection",
            "status": "connected",
            "message": "Live detection stream ready"
        })

        while True:
            if EVENT_LOG.exists():
                with EVENT_LOG.open("r", encoding="utf-8") as file:
                    file.seek(position)

                    while True:
                        line = file.readline()

                        if not line:
                            break

                        position = file.tell()
                        line = line.strip()

                        if not line:
                            continue

                        try:
                            await websocket.send_json({
                                "type": "detection",
                                "data": json.loads(line)
                            })
                        except json.JSONDecodeError:
                            continue

            await asyncio.sleep(0.1)

    except WebSocketDisconnect:
        print("[WS] Detection client disconnected")
