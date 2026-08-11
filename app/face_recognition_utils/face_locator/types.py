from dataclasses import dataclass

from numpy import ndarray


@dataclass
class FaceLocated:
    landmarks: ndarray
    confidence: float
    bbox: ndarray
    embedding: ndarray


