import numpy as np
from neglnn.layers.activation import Activation
from neglnn.utils.types import Array

class Sigmoid(Activation):
    def call(self, x: Array) -> Array:
        return 1 / (1 + np.exp(-x))
    
    def prime(self, x: Array) -> Array:
        s = self.call(x)
        return s * (1 - s)