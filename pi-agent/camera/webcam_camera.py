
import cv2
from .base_camera import BaseCamera


class WebcamCamera(BaseCamera):

    def __init__(self, camera_index=0, width=640, height=480):
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.camera = None

    def start(self):
        self.camera = cv2.VideoCapture(
            self.camera_index,
            cv2.CAP_DSHOW
        )

        if not self.camera.isOpened():
            self.camera.release()
            self.camera = cv2.VideoCapture(self.camera_index)

        if not self.camera.isOpened():
            raise RuntimeError("Unable to open webcam")

        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        return True

    def read(self):
        if self.camera is None:
            return False, None

        success, frame = self.camera.read()
        return success, frame

    def stop(self):
        if self.camera is not None:
            self.camera.release()
            self.camera = None
