import numpy as np
from neglnn.layers.layer import Layer, BackwardState
from neglnn.utils.types import Array, Shape

class Reshape(Layer):
    def __init__(self, input_shape: Shape, output_shape: Shape):
        super().__init__(input_shape, output_shape)

    def forward(self, input: Array) -> Array:
        return np.reshape(input, self.output_shape)

    def backward(self, output_gradient: Array) -> BackwardState:
        return BackwardState(np.reshape(output_gradient, self.input_shape))