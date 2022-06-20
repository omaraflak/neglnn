from abc import ABC
from neglnn.utils.types import Float, Array

class Loss(ABC):
    def call(self, true: Array, pred: Array) -> Float:
        raise NotImplementedError
    
    def prime(self, true: Array, pred: Array) -> Array:
        raise NotImplementedError