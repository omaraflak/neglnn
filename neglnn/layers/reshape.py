import numpy as np
from neglnn.layers.layer import Layer, BackwardState
from neglnn.utils.types import Array, Shape

class Reshape(Layer):
    def __init__(self, input_shape: Shape, output_shape: Shape):
        super().__init__()
        self._input_shape = input_shape
        self._output_shape = output_shape

    def forward(self, input: Array) -> Array:
        return np.reshape(input, self._output_shape)
    
    def backward(self, output_gradient: Array) -> BackwardState:
        return BackwardState(np.reshape(output_gradient, self._input_shape))
    
    def input_shape(self) -> Shape:
        return self._input_shape
    
    def output_shape(self) -> Shape:
        return self._output_shape

    def trainable(self) -> bool:
        return False