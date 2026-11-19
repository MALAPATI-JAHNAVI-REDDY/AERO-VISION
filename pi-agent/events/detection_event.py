
from datetime import datetime, timezone
from typing import Any
import time


def create_detection_event(
    detections: list[dict[str, Any]],
    frame_id: int,
    camera_source: str
) -> dict[str, Any]:

    event_time = datetime.now(timezone.utc)

    objects = []

    for detection in detections:

        objects.append({
            "track_id": int(detection["track_id"])
                if detection.get("track_id") is not None
                else None,

            "class_id": int(detection["class_id"]),
            "class_name": str(detection["class_name"]),
            "confidence": round(
                float(detection["confidence"]), 4
            ),
            "bbox": [
                int(value) for value in detection["bbox"]
            ]
        })

    return {
        "event_id": f"frame-{frame_id}-{time.time_ns()}",
        "timestamp": event_time.isoformat(),
        "timestamp_unix": event_time.timestamp(),
        "frame_id": int(frame_id),
        "camera_source": camera_source,
        "object_count": len(objects),
        "unique_objects": len({
            obj["track_id"]
            for obj in objects
            if obj["track_id"] is not None
        }),
        "objects": objects
    }
