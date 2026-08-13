from abc import ABC, abstractmethod

from numpy import ndarray

from app.face_recognition_utils.face_locator.types import FaceLocated


class FacePrivater(ABC):

    @abstractmethod
    def hide_faces(self, faces: list[FaceLocated], image: ndarray) -> ndarray:
        ...
