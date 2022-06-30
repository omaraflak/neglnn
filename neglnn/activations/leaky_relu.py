from neglnn.layers.activation import Activation
from neglnn.utils.types import Array, Float

class LeakyRelu(Activation):
    def __init__(self, alpha: Float = 0.3):
        self.alpha = alpha

    def call(self, x: Array) -> Array:
        return ((x > 0) * x) + ((x <= 0) * self.alpha * x)
    
    def prime(self, x: Array) -> Array:
        return (x > 0) + ((x <= 0) * self.alpha)