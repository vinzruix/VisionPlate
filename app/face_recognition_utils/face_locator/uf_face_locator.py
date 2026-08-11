from numpy import ndarray
from uniface import YOLOv5Face
from uniface.constants import YOLOv5FaceWeights
from app.face_recognition_utils.face_locator.base import FaceLocatorBase
from app.face_recognition_utils.face_locator.types import FaceLocated


class UFFaceLocator(FaceLocatorBase):

    def __init__(self):
        self.detector_model = YOLOv5Face(
            model_name=YOLOv5FaceWeights.YOLOV5S,
            confidence_threshold=0.6,
            nms_threshold=0.5,
            nms_mode='torchvision',
            providers=None)

    def detect(self, image: ndarray) -> list[FaceLocated]:
        faces = self.detector_model.detect(image)

        return [FaceLocated(landmarks=face.landmarks,
                             confidence=face.confidence,
                             bbox=face.bbox) for face in faces]
