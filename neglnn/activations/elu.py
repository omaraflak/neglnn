import numpy as np
from neglnn.layers.activation import Activation
from neglnn.utils.types import Array, Float

class Elu(Activation):
    def __init__(self, alpha: Float = 1.0):
        super().__init__()
        self.alpha = alpha

    def call(self, x: Array) -> Array:
        return ((x > 0) * x) + ((x <= 0) * self.alpha * (np.exp(x) - 1))
    
    def prime(self, x: Array) -> Array:
        return (x > 0) + ((x <= 0) * self.alpha * np.exp(x))