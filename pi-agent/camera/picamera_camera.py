from .base_camera import BaseCamera
import cv2


class PiCameraModule3(BaseCamera):

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None

    def start(self):
        try:
            from picamera2 import Picamera2
            from libcamera import controls
        except ImportError as exc:
            raise RuntimeError(
                "Picamera2 is unavailable. Install it on Raspberry Pi OS."
            ) from exc

        self.camera = Picamera2()

        config = self.camera.create_video_configuration(
            main={
                "size": (self.width, self.height),
                "format": "RGB888"
            },
            controls={
                "FrameRate": self.fps
            },
            buffer_count=4
        )

        self.camera.configure(config)
        self.camera.start()

        # Enable continuous autofocus on Camera Module 3.
        try:
            self.camera.set_controls({
                "AfMode": controls.AfModeEnum.Continuous
            })
            print("[CAMERA] Continuous autofocus enabled")
        except Exception as exc:
            print(f"[CAMERA] Autofocus setting unavailable: {exc}")

        print("[CAMERA] Raspberry Pi Camera Module 3 started")
        return True

    def read(self):
        if self.camera is None:
            return False, None

        try:
            rgb_frame = self.camera.capture_array("main")

            # Convert RGB to OpenCV's BGR format.
            bgr_frame = cv2.cvtColor(
                rgb_frame,
                cv2.COLOR_RGB2BGR
            )

            return True, bgr_frame

        except Exception as exc:
            print(f"[CAMERA] Capture error: {exc}")
            return False, None

    def stop(self):
        if self.camera is not None:
            self.camera.stop()
            self.camera.close()
            self.camera = None

            print("[CAMERA] Raspberry Pi camera released")
