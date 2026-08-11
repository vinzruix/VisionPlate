from abc import ABC, abstractmethod

from numpy import ndarray

from app.face_recognition_utils.face_locator.types import FaceLocated


class FaceLocatorBase(ABC):

    @abstractmethod
    def detect(self, image: ndarray) -> list[FaceLocated]:
        ...
