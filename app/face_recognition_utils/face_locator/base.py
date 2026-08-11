from abc import ABC, abstractmethod


class FaceLocatorBase(ABC):

    @abstractmethod
    def detect(self, **kwargs):
        pass
