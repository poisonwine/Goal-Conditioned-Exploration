
from abc import ABC, abstractmethod

class AbstractTeacher(ABC):

    @abstractmethod
    def sample(self, batchsize):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass