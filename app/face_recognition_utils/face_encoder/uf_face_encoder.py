from numpy import ndarray
from uniface import EdgeFace, compute_similarity
from uniface.constants import EdgeFaceWeights
from app.face_recognition_utils.face_encoder.base import FaceEncoderBase
from app.face_recognition_utils.face_encoder.types import FaceEncoded, ComparisonResult
from app.face_recognition_utils.face_locator.types import FaceLocated


class UFFaceEncoder(FaceEncoderBase):

    def __init__(self):
        super().__init__()
        self.encoder_model = EdgeFace(model_name=EdgeFaceWeights.S_GAMMA_05)

    def encode_face(self, image: ndarray, face_located: FaceLocated) -> FaceEncoded:
        encodings = self.encoder_model.get_normalized_embedding(image, face_located.landmarks)
        return FaceEncoded(face_located=face_located, encodings=encodings)

    def compare_faces(self, face_1: FaceEncoded, face_2: FaceEncoded) -> ComparisonResult:
        comparison_result = compute_similarity(face_1.encodings, face_2.encodings, normalized=True)
        return ComparisonResult(tolerance=self.tolerance,
                                confidence=float(comparison_result),
                                )
