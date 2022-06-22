import numpy as np
from neglnn.layers.activation import Activation
from neglnn.utils.types import Array

class Relu(Activation):
    def call(self, x: Array) -> Array:
        return np.maximum(x, 0)
    
    def prime(self, x: Array) -> Array:
        return np.array(x > 0).astype('int')