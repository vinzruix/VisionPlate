from numpy import ndarray
from uniface import BlurFace, Face

from app.face_recognition_utils.face_locator.types import FaceLocated
from app.face_recognition_utils.face_privater.base import FacePrivater


class UFFacePrivater(FacePrivater):

    def __init__(self):
        self.privater_model = BlurFace(method='pixelate', pixel_blocks=15)

    def hide_faces(self, faces: list[FaceLocated], image: ndarray) -> ndarray:
        image_private = self.privater_model.anonymize(image, [Face(bbox=face.bbox,
                                                                   confidence=face.confidence,
                                                                   landmarks=face.landmarks) for face in faces])

        return image_private
