import numpy as np
from neglnn.initializers.initializer import Initializer
from neglnn.utils.types import Array, Float

class RandomUniform(Initializer):
    def __init__(self, low: Float = -0.5, high: Float = 0.5):
        super().__init__()
        self.low = low
        self.high = high

    def get(self, *shape: int) -> Array:
        return np.random.uniform(self.low, self.high, shape)