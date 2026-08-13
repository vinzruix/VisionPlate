from numpy import ndarray
from app.face_recognition_utils.face_encoder.base import FaceEncoderBase
from app.face_recognition_utils.face_encoder.types import FaceEncoded, ComparisonResult
from app.face_recognition_utils.face_locator.base import FaceLocatorBase
from app.face_recognition_utils.face_locator.types import FaceLocated
from app.face_recognition_utils.face_privater.base import FacePrivater
from app.face_recognition_utils.face_spoofer.base import FaceSpooferBase
from app.face_recognition_utils.face_spoofer.types import SpoofResult


class FaceRecognitionService:

    def __init__(self, locator: FaceLocatorBase, encoder: FaceEncoderBase, spoofer: FaceSpooferBase,
                 privater: FacePrivater):
        self.locator = locator
        self.encoder = encoder
        self.spoofer = spoofer
        self.privater = privater

    def detect_faces_in_image(self, image) -> list[FaceLocated]:
        return self.locator.detect(image=image)

    def encode_face(self, image, face: FaceLocated) -> FaceEncoded:
        return self.encoder.encode_face(image=image, face_located=face)

    def compare_faces(self, face_1: FaceEncoded, face_2: FaceEncoded) -> ComparisonResult:
        return self.encoder.compare_faces(face_1, face_2)

    def verify_antispoofing(self, face: FaceLocated, image: ndarray) -> SpoofResult:
        return self.spoofer.check_spoofing(image=image, face=face)

    def anonymize_faces(self, faces: list[FaceLocated], image: ndarray) -> ndarray:
        return self.privater.hide_faces(image=image, faces=faces)
