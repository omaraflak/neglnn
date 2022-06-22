import numpy as np
from neglnn.initializers.initializer import Initializer
from neglnn.utils.types import Array, Float

class Normal(Initializer):
    def __init__(self, m: Float = 0.0, s: Float = 1.0):
        super().__init__()
        self.m = m
        self.s = s

    def get(self, *shape: int) -> Array:
        return np.random.normal(self.m, self.s, shape)