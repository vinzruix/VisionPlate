from typing import List
from numpy import ndarray
from uniface import compute_similarity
from app.face_recognition_utils.face_encoder.base import FaceEncoderBase
from app.face_recognition_utils.face_encoder.types import FaceEncoded
from app.face_recognition_utils.face_locator.base import FaceLocatorBase
from app.face_recognition_utils.face_locator.types import FaceLocated
from app.face_recognition_utils.face_spoofer.base import FaceSpooferBase


class FaceRecognitionService:

    def __init__(self, locator: FaceLocatorBase, encoder: FaceEncoderBase, spoofer: FaceSpooferBase,
                 comparison_tolerance: float = 0.5, spoofing_tolerance: float = 0.7):
        self.locator = locator
        self.encoder = encoder
        self.spoofer = spoofer
        self.comparison_tolerance = comparison_tolerance
        self.spoofing_tolerance = spoofing_tolerance

    def detect_faces_in_image(self, image) -> List[FaceLocated]:
        return self.locator.detect(image=image)

    def encode_face(self, image, face: FaceLocated) -> FaceEncoded:
        return self.encoder.encode_face(image=image, face=face)

    def compare_faces(self, face_1: FaceEncoded, face_2: FaceEncoded) -> bool:
        comparison_result = compute_similarity(face_1.encodings, face_2.encodings)
        return True if comparison_result > self.comparison_tolerance else False

    def verify_antispoofing(self, face: FaceEncoded, image: ndarray) -> bool:
        spoof_result = self.spoofer.check_spoofing(image=image, face=face)
        return True if spoof_result.is_real and spoof_result.confidence >= self.spoofing_tolerance else False
