from abc import ABC, abstractmethod

from numpy import ndarray

from app.face_recognition_utils.face_locator.types import FaceLocated
from app.face_recognition_utils.face_spoofer.types import SpoofResult


class FaceSpooferBase(ABC):

    @abstractmethod
    def check_spoofing(self, image: ndarray, face: FaceLocated) -> SpoofResult:
        ...
