from typing import List

from uniface import compute_similarity

from app.face_recognition_utils.face_encoder.base import FaceEncoderBase
from app.face_recognition_utils.face_locator.base import FaceLocatorBase
from app.face_recognition_utils.types import FaceEncoded, FaceLocated


class FaceRecognitionService:

    def __init__(self, locator: FaceLocatorBase, encoder: FaceEncoderBase, tolerance: float = 0.5):
        self.locator = locator
        self.encoder = encoder
        self.tolerance = tolerance

    def detect_faces_in_image(self, image) -> List[FaceLocated]:
        return self.locator.detect(image=image)

    def encode_face(self, image, face: FaceLocated) -> FaceEncoded:
        return self.encoder.encode_face(image=image, face=face)

    def compare_faces(self, face_1: FaceEncoded, face_2: FaceEncoded) -> bool:
        comparison_result = compute_similarity(face_1.encodings, face_2.encodings)

        return True if comparison_result > self.tolerance else False
