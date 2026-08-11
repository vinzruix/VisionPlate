from numpy import ndarray
from uniface import MiniFASNet
from app.face_recognition_utils.face_locator.types import FaceLocated
from app.face_recognition_utils.face_spoofer.base import FaceSpooferBase
from app.face_recognition_utils.face_spoofer.types import SpoofResult


class UFFaceSpoofer(FaceSpooferBase):
    def __init__(self):
        super().__init__()
        self.spoofer_model = MiniFASNet()

    def check_spoofing(self, image: ndarray, face: FaceLocated) -> SpoofResult:
        result = self.spoofer_model.predict(image=image, bbox=face.bbox)
        return SpoofResult(confidence=result.confidence,
                           face_located=face,
                           is_real=result.is_real,
                           tolerance=self.tolerance)
