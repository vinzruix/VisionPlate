from dataclasses import dataclass

from numpy import ndarray

from app.face_recognition_utils.face_locator.types import FaceLocated


@dataclass
class FaceEncoded:
    face_located: FaceLocated
    encodings: ndarray
