import numpy as np
from neglnn.initializers.initializer import Initializer
from neglnn.utils.types import Array

class Normal(Initializer):
    def get(self, *shape: int) -> Array:
        return np.random.rand(*shape) - 0.5