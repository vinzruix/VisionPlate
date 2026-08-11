from dataclasses import dataclass

from app.face_recognition_utils.face_locator.types import FaceLocated


@dataclass(frozen=True)
class SpoofResult:
    tolerance : float
    face_located: FaceLocated
    is_real: bool
    confidence: float

    @property
    def is_spoofing(self):
        return self.is_real and self.confidence >= self.tolerance
