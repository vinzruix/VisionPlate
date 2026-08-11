from numpy import ndarray
from uniface import EdgeFace, compute_similarity
from uniface.constants import EdgeFaceWeights
from app.face_recognition_utils.face_encoder.base import FaceEncoderBase
from app.face_recognition_utils.face_encoder.types import FaceEncoded
from app.face_recognition_utils.face_locator.types import FaceLocated


class UFFaceEncoder(FaceEncoderBase):

    def __init__(self, comparison_tolerance):
        self.encoder_model = EdgeFace(model_name=EdgeFaceWeights.S_GAMMA_05)
        self.comparison_tolerance = comparison_tolerance

    def encode_face(self, image: ndarray, face_located: FaceLocated) -> FaceEncoded:
        encodings = self.encoder_model.get_normalized_embedding(image, face_located.landmarks)
        return FaceEncoded(face_located=face_located, encodings=encodings)

    def compare_faces(self, face_1: FaceEncoded, face_2: FaceEncoded) -> bool:
        comparison_result = compute_similarity(face_1.encodings, face_2.encodings)
        return comparison_result > self.comparison_tolerance
