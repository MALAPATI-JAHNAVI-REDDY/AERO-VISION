
import sys
import json
import threading
import time

from pathlib import Path

import cv2

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PI_AGENT_DIR = PROJECT_ROOT / "pi-agent"

if str(PI_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(PI_AGENT_DIR))

# Project modules
from inference.detector import DroneVisionDetector
from inference.tracker import IoUTracker
from camera.webcam_camera import WebcamCamera
from camera.picamera_camera import PiCameraModule3
from events.detection_event import create_detection_event


CONFIG_PATH = PI_AGENT_DIR / "config" / "model_config.yaml"
EVENT_LOG = PROJECT_ROOT / "logs" / "detections.jsonl"


class CameraService:

    def __init__(self, camera_type="webcam"):
        self.camera_type = camera_type

        self.camera = None
        self.detector = None

        # Video frames
        self.raw_frame = None
        self.annotated_frame = None

        # Detection and tracking
        self.detections = []

        self.tracker = IoUTracker(
            iou_threshold=0.3,
            max_age=10
        )

        # Service state
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Performance metrics
        self.frame_id = 0
        self.fps = 0.0
        self.inference_ms = 0.0

        # Tracking statistics
        self.visible_objects = 0
        self.tracked_objects = 0
        self.unique_objects = 0

    def start(self):

        if self.running:
            return

        print("[CAMERA SERVICE] Initializing...")

        self.detector = DroneVisionDetector(CONFIG_PATH)

        # Select camera
        if self.camera_type == "picamera":
            self.camera = PiCameraModule3(
                width=640,
                height=480,
                fps=30
            )
        else:
            self.camera = WebcamCamera(
                camera_index=0,
                width=640,
                height=480
            )

        self.camera.start()

        # Reset session state
        self.tracker.reset()

        with self.lock:
            self.frame_id = 0
            self.fps = 0.0
            self.inference_ms = 0.0
            self.detections = []
            self.visible_objects = 0
            self.tracked_objects = 0
            self.unique_objects = 0
            self.raw_frame = None
            self.annotated_frame = None

        self.running = True

        self.thread = threading.Thread(
            target=self._capture_loop,
            daemon=True
        )

        self.thread.start()

        print(f"[CAMERA SERVICE] Started: {self.camera_type}")

    def _capture_loop(self):

        previous_time = time.perf_counter()

        try:

            while self.running:

                success, frame = self.camera.read()

                if not success:
                    time.sleep(0.01)
                    continue

                self.frame_id += 1

                # Run inference every second frame
                if self.frame_id % 2 == 1:

                    inference_start = time.perf_counter()

                    detections = self.detector.detect(frame)

                    inference_ms = (
                        time.perf_counter() - inference_start
                    ) * 1000

                    # Assign persistent tracking IDs
                    tracked_detections = self.tracker.update(
                        detections
                    )

                    # Calculate tracking statistics
                    visible_count = len(tracked_detections)
                    tracked_count = len(self.tracker.tracks)

                    unique_count = len({
                        detection.get("track_id")
                        for detection in tracked_detections
                        if detection.get("track_id") is not None
                    })

                    # Create detection event
                    event = create_detection_event(
                        detections=tracked_detections,
                        frame_id=self.frame_id,
                        camera_source=self.camera_type
                    )

                    # Add separate tracking statistics
                    event["visible_objects"] = visible_count
                    event["tracked_objects"] = tracked_count
                    event["unique_objects"] = unique_count

                    # Save detection state
                    with self.lock:
                        self.detections = tracked_detections
                        self.inference_ms = inference_ms
                        self.visible_objects = visible_count
                        self.tracked_objects = tracked_count
                        self.unique_objects = unique_count

                    # Save event to JSONL log
                    EVENT_LOG.parent.mkdir(
                        parents=True,
                        exist_ok=True
                    )

                    with EVENT_LOG.open(
                        "a",
                        encoding="utf-8"
                    ) as log_file:

                        log_file.write(
                            json.dumps(
                                event,
                                ensure_ascii=False
                            ) + "\n"
                        )

                # Get current detections safely
                with self.lock:
                    current_detections = list(self.detections)

                # Annotate frame
                annotated = self.detector.annotate(
                    frame,
                    current_detections
                )

                # Calculate display FPS
                current_time = time.perf_counter()
                elapsed = current_time - previous_time

                if elapsed > 0:

                    instant_fps = 1.0 / elapsed

                    self.fps = (
                        instant_fps
                        if self.fps == 0
                        else 0.9 * self.fps + 0.1 * instant_fps
                    )

                previous_time = current_time

                # Publish frames safely
                with self.lock:
                    self.raw_frame = frame.copy()
                    self.annotated_frame = annotated.copy()

        except Exception as exc:

            print(f"[CAMERA SERVICE] Error: {exc}")
            self.running = False

        finally:

            if self.camera:
                self.camera.stop()

            print("[CAMERA SERVICE] Capture loop ended")

    def get_frame(self, annotated=False):

        with self.lock:

            frame = (
                self.annotated_frame
                if annotated
                else self.raw_frame
            )

            if frame is None:
                return None

            return frame.copy()

    def get_metrics(self):

        with self.lock:

            return {
                "running": self.running,
                "camera_source": self.camera_type,
                "frame_id": self.frame_id,
                "fps": round(self.fps, 2),
                "inference_ms": round(self.inference_ms, 2),
                "unique_objects": self.unique_objects,
                "visible_objects": self.visible_objects,
                "tracked_objects": self.tracked_objects,
                "detections": len(self.detections)
            }

    def stop(self):

        self.running = False

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=3)

        print("[CAMERA SERVICE] Stopped")


# Shared camera service instance
camera_service = CameraService(
    camera_type="webcam"
)
