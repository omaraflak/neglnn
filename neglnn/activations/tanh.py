import numpy as np
from neglnn.layers.activation import Activation
from neglnn.utils.types import Array

class Tanh(Activation):
    def call(self, x: Array) -> Array:
        return np.tanh(x)
    
    def prime(self, x: Array) -> Array:
        return 1 - np.power(np.tanh(x), 2)