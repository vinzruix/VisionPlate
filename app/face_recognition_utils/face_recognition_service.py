from numpy import ndarray
from uniface import compute_similarity
from app.face_recognition_utils.face_encoder.base import FaceEncoderBase
from app.face_recognition_utils.face_encoder.types import FaceEncoded
from app.face_recognition_utils.face_locator.base import FaceLocatorBase
from app.face_recognition_utils.face_locator.types import FaceLocated
from app.face_recognition_utils.face_spoofer.base import FaceSpooferBase


class FaceRecognitionService:

    def __init__(self, locator: FaceLocatorBase, encoder: FaceEncoderBase, spoofer: FaceSpooferBase,
                 spoofing_tolerance: float = 0.7):
        self.locator = locator
        self.encoder = encoder
        self.spoofer = spoofer
        self.spoofing_tolerance = spoofing_tolerance

    def detect_faces_in_image(self, image) -> list[FaceLocated]:
        return self.locator.detect(image=image)

    def encode_face(self, image, face: FaceLocated) -> FaceEncoded:
        return self.encoder.encode_face(image=image, face_located=face)

    def compare_faces(self, face_1: FaceEncoded, face_2: FaceEncoded) -> bool:
        return self.encoder.compare_faces(face_1, face_2)

    def verify_antispoofing(self, face: FaceEncoded, image: ndarray) -> bool:
        spoof_result = self.spoofer.check_spoofing(image=image, face=face.face_located)
        return spoof_result.is_real and spoof_result.confidence >= self.spoofing_tolerance
