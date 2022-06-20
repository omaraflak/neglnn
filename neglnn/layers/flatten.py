import numpy as np
from neglnn.layers.reshape import Reshape
from neglnn.utils.types import Shape

class Flatten(Reshape):
    def __init__(self, input_shape: Shape):
        super().__init__(input_shape, (1, np.prod(input_shape)))