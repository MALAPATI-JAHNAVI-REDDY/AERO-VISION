
import numpy as np


def calculate_iou(box, boxes):
    """
    Calculate Intersection over Union (IoU).

    Boxes use XYXY coordinates:
    [x1, y1, x2, y2]
    """

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])

    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = (
        np.maximum(0, x2 - x1) *
        np.maximum(0, y2 - y1)
    )

    area_a = (
        max(0, box[2] - box[0]) *
        max(0, box[3] - box[1])
    )

    area_b = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0]) *
        np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    union = area_a + area_b - intersection

    return intersection / np.maximum(union, 1e-7)


def class_aware_nms(boxes, scores, class_ids, threshold):
    """
    Apply Non-Maximum Suppression separately
    for each object class.
    """

    keep = []

    for class_id in np.unique(class_ids):

        indices = np.where(class_ids == class_id)[0]

        indices = indices[
            np.argsort(scores[indices])[::-1]
        ]

        while len(indices) > 0:

            current = indices[0]
            keep.append(current)

            if len(indices) == 1:
                break

            remaining = indices[1:]

            ious = calculate_iou(
                boxes[current],
                boxes[remaining]
            )

            indices = remaining[ious <= threshold]

    return sorted(
        keep,
        key=lambda index: scores[index],
        reverse=True
    )


def decode_predictions(
    output,
    image_info,
    confidence_threshold=0.25,
    nms_threshold=0.45,
    max_detections=100
):
    """
    Decode standard YOLOv5 output.

    Expected:
        Shape: [1, 25200, 85]
        Values: xywh + objectness + 80 class probabilities

    Confidence:
        objectness * class probability
    """

    predictions = np.asarray(output)

    if predictions.ndim == 3:
        predictions = predictions[0]

    if predictions.ndim != 2 or predictions.shape[1] != 85:
        raise ValueError(
            f"Unexpected output shape: {predictions.shape}"
        )

    predictions = predictions.astype(np.float32)

    if not np.isfinite(predictions).all():
        raise ValueError("Model output contains invalid values")

    class_scores = predictions[:, 5:85]

    class_ids = np.argmax(
        class_scores,
        axis=1
    )

    best_class_scores = class_scores[
        np.arange(len(predictions)),
        class_ids
    ]

    scores = predictions[:, 4] * best_class_scores

    valid = scores >= confidence_threshold

    predictions = predictions[valid]
    scores = scores[valid]
    class_ids = class_ids[valid]

    if len(predictions) == 0:
        return []

    # Convert XYWH into XYXY.
    centers = predictions[:, :2]
    sizes = predictions[:, 2:4]

    boxes = np.concatenate(
        [
            centers - sizes / 2,
            centers + sizes / 2
        ],
        axis=1
    )

    # Reverse letterbox transformation.
    scale = image_info["scale"]
    left = image_info["left"]
    top = image_info["top"]

    boxes[:, [0, 2]] = (
        boxes[:, [0, 2]] - left
    ) / scale

    boxes[:, [1, 3]] = (
        boxes[:, [1, 3]] - top
    ) / scale

    # Clip boxes to original image dimensions.
    width = image_info["original_width"]
    height = image_info["original_height"]

    boxes[:, [0, 2]] = np.clip(
        boxes[:, [0, 2]],
        0,
        width
    )

    boxes[:, [1, 3]] = np.clip(
        boxes[:, [1, 3]],
        0,
        height
    )

    valid_boxes = (
        (boxes[:, 2] > boxes[:, 0]) &
        (boxes[:, 3] > boxes[:, 1])
    )

    boxes = boxes[valid_boxes]
    scores = scores[valid_boxes]
    class_ids = class_ids[valid_boxes]

    if len(boxes) == 0:
        return []

    # Class-aware NMS.
    keep = class_aware_nms(
        boxes,
        scores,
        class_ids,
        nms_threshold
    )

    detections = []

    for index in keep[:max_detections]:

        x1, y1, x2, y2 = boxes[index]

        detections.append({
            "class_id": int(class_ids[index]),
            "confidence": float(scores[index]),
            "bbox": [
                int(x1),
                int(y1),
                int(x2),
                int(y2)
            ]
        })

    return detections
