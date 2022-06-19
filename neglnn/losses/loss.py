from neglnn.network.state import Stateful
from neglnn.utils.types import Float, Array

class Loss(Stateful):
    def call(self, true: Array, pred: Array) -> Float:
        raise NotImplementedError
    
    def prime(self, true: Array, pred: Array) -> Array:
        raise NotImplementedError