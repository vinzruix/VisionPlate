from dataclasses import dataclass
from typing import Optional

from numpy import ndarray


@dataclass
class FaceLocated:
    landmarks: ndarray
    confidence: float
    bbox: ndarray
    embedding: ndarray


@dataclass
class FaceEncoded:
    face_located: FaceLocated
    encodings: ndarray
