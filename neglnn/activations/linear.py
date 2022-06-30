import numpy as np
from neglnn.layers.activation import Activation
from neglnn.utils.types import Array, Float

class Linear(Activation):
    def __init__(self, c: Float = 1.0):
        super().__init__()
        self.c = c

    def call(self, x: Array) -> Array:
        return self.c * x
    
    def prime(self, x: Array) -> Array:
        return np.full_like(x, self.c)