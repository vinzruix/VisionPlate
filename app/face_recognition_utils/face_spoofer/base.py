from abc import ABC

from app.face_recognition_utils.face_spoofer.types import SpoofResult


class FaceSpooferBase(ABC):

    def check_spoofing(self, **kwargs) -> SpoofResult:
        pass
