from abc import ABC, abstractmethod
from typing import List

from app.face_recognition_utils.types import FaceLocated


class FaceLocatorBase(ABC):

    @abstractmethod
    def detect(self, **kwargs) -> List[FaceLocated]:
        pass
