from abc import ABC, abstractmethod


class FaceEncoderBase(ABC):

    @abstractmethod
    def encode_face(self, **kwargs):
        pass
