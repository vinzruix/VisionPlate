from numpy import ndarray
from uniface import EdgeFace, Face
from uniface.constants import EdgeFaceWeights
from app.face_recognition_utils.face_encoder.base import FaceEncoderBase
from app.face_recognition_utils.face_encoder.types import FaceEncoded
from app.face_recognition_utils.face_locator.types import FaceLocated


class UFFaceEncoder(FaceEncoderBase):

    def __init__(self):
        self.encoder_model = EdgeFace(model_name=EdgeFaceWeights.S_GAMMA_05)

    def encode_face(self, image: ndarray, face_located: FaceLocated) -> FaceEncoded:
        encodings = self.encoder_model.get_normalized_embedding(image, face_located.landmarks)
        return FaceEncoded(face_located=face_located, encodings=encodings)
