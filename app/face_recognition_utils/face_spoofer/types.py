from dataclasses import dataclass

from app.face_recognition_utils.face_locator.types import FaceLocated


@dataclass
class SpoofResult:
    face_located: FaceLocated
    is_real: bool
    confidence: float
