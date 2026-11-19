
from pathlib import Path

import cv2
import yaml

from .model import YOLOv5Model
from .preprocessing import preprocess_image
from .postprocessing import decode_predictions


COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana",
    "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


class DroneVisionDetector:

    def __init__(self, config_path):

        config_path = Path(config_path).resolve()

        project_root = config_path.parents[2]

        with open(config_path, "r", encoding="utf-8") as file:
            self.config = yaml.safe_load(file)

        model_path = (
            project_root /
            self.config["model"]["path"]
        )

        self.input_size = self.config["model"]["input_size"]

        detection_config = self.config["detection"]

        self.confidence_threshold = detection_config[
            "confidence_threshold"
        ]

        self.nms_threshold = detection_config[
            "nms_threshold"
        ]

        self.max_detections = detection_config[
            "max_detections"
        ]

        self.model = YOLOv5Model(
            model_path,
            threads=self.config["runtime"]["threads"]
        )

        print("[DETECTOR] Drone Vision initialized")

    def detect(self, image):

        if image is None or image.size == 0:
            raise ValueError("Invalid input image")

        tensor, image_info = preprocess_image(
            image,
            self.input_size
        )

        output = self.model.infer(tensor)

        detections = decode_predictions(
            output,
            image_info,
            confidence_threshold=self.confidence_threshold,
            nms_threshold=self.nms_threshold,
            max_detections=self.max_detections
        )

        for detection in detections:

            detection["class_name"] = COCO_CLASSES[
                detection["class_id"]
            ]

        return detections

    
    def annotate(self, image, detections):

        result = image.copy()

        for detection in detections:

            x1, y1, x2, y2 = detection["bbox"]

            class_name = detection["class_name"]
            confidence = detection["confidence"]
            track_id = detection.get("track_id")

            if track_id is not None:
                label = (
                    f"ID {track_id} | {class_name} "
                    f"{confidence:.0%}"
                )
            else:
                label = f"{class_name} {confidence:.0%}"

            cv2.rectangle(
                result,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            cv2.putText(
                result,
                label,
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2
            )

        return result
