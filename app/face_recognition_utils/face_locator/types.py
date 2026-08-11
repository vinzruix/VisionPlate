from dataclasses import dataclass

from numpy import ndarray


@dataclass(frozen=True)
class FaceLocated:
    landmarks: ndarray
    confidence: float
    bbox: ndarray


