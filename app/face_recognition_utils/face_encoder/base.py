from abc import ABC, abstractmethod
from numpy import ndarray
from app.face_recognition_utils.face_encoder.types import FaceEncoded
from app.face_recognition_utils.face_locator.types import FaceLocated


class FaceEncoderBase(ABC):

    @abstractmethod
    def encode_face(self, image: ndarray, face_located: FaceLocated) -> FaceEncoded:
        ...

    @abstractmethod
    def compare_faces(self, face_1: FaceEncoded, face_2: FaceEncoded) -> bool:
        ...

