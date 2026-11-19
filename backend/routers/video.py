
import base64
import sys
import threading
import time
import uuid

from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PI_AGENT_DIR = PROJECT_ROOT / "pi-agent"

if str(PI_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(PI_AGENT_DIR))

from inference.detector import DroneVisionDetector

from backend.services.camera_service import camera_service


# --------------------------------------------------
# ROUTER AND CONFIGURATION
# --------------------------------------------------

router = APIRouter(tags=["Video"])

MEDIA_DIR = PROJECT_ROOT / "generated_media"
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_PATH = PI_AGENT_DIR / "config" / "model_config.yaml"

MAX_IMAGE_SIZE = 10 * 1024 * 1024
MAX_VIDEO_SIZE = 50 * 1024 * 1024

# --------------------------------------------------
# DETECTOR AND LOCKS
# --------------------------------------------------

media_detector = None

detector_lock = threading.Lock()
media_lock = threading.Lock()

# Uploaded video sessions for this backend process.
media_jobs = {}


def get_media_detector():
    global media_detector

    with detector_lock:
        if media_detector is None:
            media_detector = DroneVisionDetector(CONFIG_PATH)

        return media_detector


# --------------------------------------------------
# DETECTION SERIALIZATION
# --------------------------------------------------

def serialize_detections(detections):

    objects = []

    for detection in detections:

        objects.append({
            "track_id": (
                int(detection["track_id"])
                if detection.get("track_id") is not None
                else None
            ),

            "class_id": int(detection["class_id"]),

            "class_name": str(
                detection["class_name"]
            ),

            "confidence": round(
                float(detection["confidence"]),
                4
            ),

            "bbox": [
                int(value)
                for value in detection["bbox"]
            ]
        })

    return objects


def make_event(
    objects,
    source,
    filename,
    frame_index=None,
    video_timestamp=None
):

    now = datetime.now(timezone.utc)

    return {
        "event_id": f"{source}-{uuid.uuid4().hex}",

        "source": source,

        "filename": filename,

        "timestamp": now.isoformat(),

        "timestamp_unix": now.timestamp(),

        "frame_id": frame_index,

        "video_timestamp": video_timestamp,

        "object_count": len(objects),

        "objects": objects
    }


# --------------------------------------------------
# LIVE CAMERA STREAMING
# --------------------------------------------------

def generate_live_frames(annotated=False):

    while camera_service.running:

        frame = camera_service.get_frame(
            annotated=annotated
        )

        if frame is None:
            time.sleep(0.05)
            continue

        success, buffer = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 80]
        )

        if not success:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n"
            + buffer.tobytes()
            + b"\r\n"
        )


@router.get("/video/raw")
def raw_video():

    if not camera_service.running:
        raise HTTPException(
            status_code=503,
            detail="Live camera is not running."
        )

    return StreamingResponse(
        generate_live_frames(False),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@router.get("/video/annotated")
def annotated_video():

    if not camera_service.running:
        raise HTTPException(
            status_code=503,
            detail="Live camera is not running."
        )

    return StreamingResponse(
        generate_live_frames(True),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# --------------------------------------------------
# UPLOAD IMAGE OR VIDEO
# --------------------------------------------------

@router.post("/media/upload")
async def upload_media(
    file: UploadFile = File(...)
):

    filename = Path(
        file.filename or "upload"
    ).name

    extension = Path(filename).suffix.lower()

    is_image = extension in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp"
    }

    is_video = extension == ".mp4"

    if not is_image and not is_video:

        raise HTTPException(
            status_code=400,
            detail="Only JPG, JPEG, PNG, WEBP and MP4 files are supported."
        )

    max_size = (
        MAX_IMAGE_SIZE
        if is_image
        else MAX_VIDEO_SIZE
    )

    contents = await file.read(max_size + 1)

    if not contents:

        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty."
        )

    if len(contents) > max_size:

        raise HTTPException(
            status_code=413,
            detail=f"Maximum file size is {max_size // (1024 * 1024)} MB."
        )

    # ----------------------------------------------
    # IMAGE ANALYSIS
    # ----------------------------------------------

    if is_image:

        with media_lock:

            detector = get_media_detector()

            image = cv2.imdecode(
                np.frombuffer(
                    contents,
                    dtype=np.uint8
                ),
                cv2.IMREAD_COLOR
            )

            if image is None:

                raise HTTPException(
                    status_code=400,
                    detail="Could not decode this image."
                )

            detections = detector.detect(image)

            annotated = detector.annotate(
                image,
                detections
            )

            success, encoded = cv2.imencode(
                ".jpg",
                annotated
            )

            if not success:

                raise HTTPException(
                    status_code=500,
                    detail="Could not encode the annotated image."
                )

            objects = serialize_detections(
                detections
            )

            event = make_event(
                objects=objects,
                source="uploaded_image",
                filename=filename
            )

            return {
                "status": "success",

                "media_type": "image",

                "filename": filename,

                "width": int(image.shape[1]),

                "height": int(image.shape[0]),

                "detections": objects,

                "object_count": len(objects),

                "annotated_image": (
                    "data:image/jpeg;base64,"
                    + base64.b64encode(
                        encoded.tobytes()
                    ).decode("utf-8")
                ),

                "activity_events": [event]
            }

    # ----------------------------------------------
    # VIDEO UPLOAD
    # Store original video; do not process all frames.
    # ----------------------------------------------

    media_id = uuid.uuid4().hex

    video_path = MEDIA_DIR / f"{media_id}.mp4"

    video_path.write_bytes(contents)

    capture = cv2.VideoCapture(
        str(video_path)
    )

    if not capture.isOpened():

        capture.release()

        video_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="Could not open this MP4 video."
        )

    fps = capture.get(
        cv2.CAP_PROP_FPS
    )

    frame_count = int(
        capture.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    width = int(
        capture.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        capture.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    capture.release()

    if (
        fps <= 0
        or frame_count <= 0
        or width <= 0
        or height <= 0
    ):

        video_path.unlink(missing_ok=True)

        raise HTTPException(
            status_code=400,
            detail="The MP4 has invalid video metadata."
        )

    duration = frame_count / fps

    media_jobs[media_id] = {
        "path": video_path,

        "filename": filename,

        "fps": fps,

        "frame_count": frame_count,

        "width": width,

        "height": height,

        "duration": duration
    }

    return {
        "status": "success",

        "media_type": "video",

        "media_id": media_id,

        "filename": filename,

        "width": width,

        "height": height,

        "fps": fps,

        "total_frames": frame_count,

        "duration_seconds": duration
    }


# --------------------------------------------------
# ANALYSE ONE EXACT VIDEO FRAME
# --------------------------------------------------

@router.get("/media/{media_id}/frame")
def analyze_video_frame(
    media_id: str,
    frame_index: int
):

    job = media_jobs.get(media_id)

    if not job:

        raise HTTPException(
            status_code=404,
            detail="Video session not found. Upload the video again."
        )

    if (
        frame_index < 0
        or frame_index >= job["frame_count"]
    ):

        raise HTTPException(
            status_code=400,
            detail="Frame index is outside the video."
        )

    with media_lock:

        capture = cv2.VideoCapture(
            str(job["path"])
        )

        if not capture.isOpened():

            raise HTTPException(
                status_code=500,
                detail="Could not read the uploaded video."
            )

        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_index
        )

        success, frame = capture.read()

        capture.release()

        if not success or frame is None:

            raise HTTPException(
                status_code=422,
                detail="Could not decode the requested frame."
            )

        detector = get_media_detector()

        detections = detector.detect(frame)

        annotated = detector.annotate(
            frame,
            detections
        )

        encoded_ok, encoded = cv2.imencode(
            ".jpg",
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 85]
        )

        if not encoded_ok:

            raise HTTPException(
                status_code=500,
                detail="Could not encode the annotated frame."
            )

        objects = serialize_detections(
            detections
        )

        video_timestamp = (
            frame_index / job["fps"]
        )

        event = make_event(
            objects=objects,
            source="uploaded_video",
            filename=job["filename"],
            frame_index=frame_index,
            video_timestamp=video_timestamp
        )

        return {
            "status": "success",

            "frame_index": frame_index,

            "video_timestamp": video_timestamp,

            "timestamp": event["timestamp"],

            "object_count": len(objects),

            "detections": objects,

            "annotated_image": (
                "data:image/jpeg;base64,"
                + base64.b64encode(
                    encoded.tobytes()
                ).decode("utf-8")
            ),

            "activity_event": event
        }
