import json
from events.detection_event import create_detection_event

import argparse
import time
import cv2

from pathlib import Path
from inference.detector import DroneVisionDetector
from camera.webcam_camera import WebcamCamera
from camera.picamera_camera import PiCameraModule3


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "config" / "model_config.yaml"

EVENTS_DIR = PROJECT_ROOT / "logs"
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

EVENT_LOG = EVENTS_DIR / "detections.jsonl"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--camera",
        choices=["webcam", "picamera"],
        default="webcam"
    )
    args = parser.parse_args()

    print("\n========== DRONE VISION LIVE SYSTEM ==========\n")

    detector = DroneVisionDetector(CONFIG_PATH)

    if args.camera == "picamera":
        camera = PiCameraModule3(width=640, height=480, fps=30)
    else:
        camera = WebcamCamera(camera_index=0, width=640, height=480)

    camera.start()

    print(f"[CAMERA] Source: {args.camera}")
    print("[SYSTEM] Press Q to quit")

    frame_count = 0
    detections = []
    fps = 0.0
    inference_ms = 0.0
    previous_time = time.perf_counter()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("[CAMERA] Frame capture failed")
                break

            frame_count += 1

            if frame_count % 2 == 1:
                inference_start = time.perf_counter()

                detections = detector.detect(frame)

                inference_ms = (
                    time.perf_counter() - inference_start
                ) * 1000

                event = create_detection_event(
                    detections=detections,
                    frame_id=frame_count,
                    camera_source=args.camera
                )

                with open(EVENT_LOG, "a", encoding="utf-8") as log_file:
                    log_file.write(
                        json.dumps(event, ensure_ascii=False) + "\n"
                    )

            display_frame = detector.annotate(frame, detections)

            current_time = time.perf_counter()
            elapsed = current_time - previous_time

            if elapsed > 0:
                instant_fps = 1.0 / elapsed
                fps = instant_fps if fps == 0 else (
                    0.9 * fps + 0.1 * instant_fps
                )

            previous_time = current_time

            cv2.putText(
                display_frame,
                f"Display FPS: {fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"Inference: {inference_ms:.1f} ms",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.putText(
                display_frame,
                f"Objects: {len(detections)}",
                (15, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0),
                2
            )

            cv2.imshow("DRONE VISION", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.stop()

        try:
            cv2.destroyAllWindows()
        except cv2.error as exc:
            print(f"[CAMERA] Window cleanup warning: {exc}")

        print("\n[SYSTEM] Camera stopped")
        print("[SYSTEM] Live detection ended")


if __name__ == "__main__":
    main()
