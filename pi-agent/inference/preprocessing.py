
import cv2
import numpy as np


def letterbox(image, target_size=640, color=(114, 114, 114)):
    """
    Resize an image while preserving aspect ratio.
    Add padding to produce a square image.
    """

    original_height, original_width = image.shape[:2]

    scale = min(
        target_size / original_width,
        target_size / original_height
    )

    new_width = int(round(original_width * scale))
    new_height = int(round(original_height * scale))

    resized = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_LINEAR
    )

    pad_width = target_size - new_width
    pad_height = target_size - new_height

    left = int(round(pad_width / 2 - 0.1))
    right = int(round(pad_width / 2 + 0.1))

    top = int(round(pad_height / 2 - 0.1))
    bottom = int(round(pad_height / 2 + 0.1))

    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color
    )

    return padded, scale, left, top


def preprocess_image(image, input_size=640):
    """
    Convert OpenCV BGR image into YOLOv5n input tensor.

    Output:
        Tensor shape: [1, 3, 640, 640]
        Data type: float16
    """

    if image is None or image.size == 0:
        raise ValueError("Invalid or empty input image")

    padded, scale, left, top = letterbox(
        image,
        input_size
    )

    rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB
    )

    tensor = rgb.astype(np.float16) / np.float16(255.0)

    tensor = np.transpose(
        tensor,
        (2, 0, 1)
    )

    tensor = np.expand_dims(
        tensor,
        axis=0
    )

    tensor = np.ascontiguousarray(tensor)

    image_info = {
        "scale": scale,
        "left": left,
        "top": top,
        "original_width": image.shape[1],
        "original_height": image.shape[0]
    }

    return tensor, image_info
