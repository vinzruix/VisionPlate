from abc import ABC, abstractmethod
from numpy import ndarray
from app.face_recognition_utils.face_encoder.types import FaceEncoded, ComparisonResult
from app.face_recognition_utils.face_locator.types import FaceLocated


class FaceEncoderBase(ABC):

    def __init__(self, tolerance: float = 0.5):
        self.tolerance = tolerance

    @abstractmethod
    def encode_face(self, image: ndarray, face_located: FaceLocated) -> FaceEncoded:
        ...

    @abstractmethod
    def compare_faces(self, face_1: FaceEncoded, face_2: FaceEncoded) -> ComparisonResult:
        ...
