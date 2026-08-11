from numpy import ndarray
from uniface import EdgeFace, Face
from uniface.constants import EdgeFaceWeights
from app.face_recognition_utils.face_encoder.base import FaceEncoderBase


class UFFaceEncoder(FaceEncoderBase):

    def __init__(self):
        self.encoder_model = EdgeFace(model_name=EdgeFaceWeights.S_GAMMA_05)

    def encode_face(self, image: ndarray, face: Face) -> ndarray:
        return self.encoder_model.get_normalized_embedding(image, face.landmarks)
