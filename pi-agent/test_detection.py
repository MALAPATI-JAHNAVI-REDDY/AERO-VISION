
from pathlib import Path
import time

import cv2

from inference.detector import DroneVisionDetector


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    Path(__file__).resolve().parent
    / "config"
    / "model_config.yaml"
)

IMAGE_PATH = PROJECT_ROOT / "sample-images" / "test.jpg"

OUTPUT_PATH = PROJECT_ROOT / "sample-images" / "annotated.jpg"


def main():

    print("\n========== DRONE VISION TEST ==========\n")

    if not IMAGE_PATH.exists():
        raise FileNotFoundError(
            f"Test image not found: {IMAGE_PATH}"
        )

    image = cv2.imread(str(IMAGE_PATH))

    if image is None:
        raise ValueError("Unable to read test image")

    print("Image dimensions:", image.shape)

    detector = DroneVisionDetector(CONFIG_PATH)

    # Warm-up
    print("\nRunning warm-up inference...")
    detector.detect(image)

    # Actual inference
    print("\nRunning object detection...")

    start = time.perf_counter()

    detections = detector.detect(image)

    elapsed = time.perf_counter() - start

    print("\nDETECTED OBJECTS")

    for item in detections:
        print(
            f'{item["class_name"]} | '
            f'{item["confidence"]:.2%} | '
            f'{item["bbox"]}'
        )

    print("\nTotal detections:", len(detections))
    print(f"Inference time: {elapsed * 1000:.2f} ms")
    print(f"Inference FPS: {1 / elapsed:.2f}")

    annotated = detector.annotate(image, detections)

    if not cv2.imwrite(str(OUTPUT_PATH), annotated):
        raise RuntimeError("Failed to save annotated image")

    print("\nAnnotated image:", OUTPUT_PATH)

    print("\n========== TEST COMPLETE ==========\n")


if __name__ == "__main__":
    main()
