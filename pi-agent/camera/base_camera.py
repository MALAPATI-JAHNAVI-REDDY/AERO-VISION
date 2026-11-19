
from abc import ABC, abstractmethod
import numpy as np


class BaseCamera(ABC):

    @abstractmethod
    def start(self):
        """Initialize and start the camera."""
        pass

    @abstractmethod
    def read(self):
        """Return (success, frame). Frame must be an OpenCV BGR image."""
        pass

    @abstractmethod
    def stop(self):
        """Release camera resources."""
        pass
