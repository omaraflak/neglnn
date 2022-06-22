import numpy as np
from typing import Optional
from neglnn.layers.layer import Layer, BackwardState
from neglnn.initializers.initializer import Initializer
from neglnn.utils.types import Array, Shape

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weights: Optional[Array] = None
        self.bias: Optional[Array] = None
    
    def on_initializer(self, initializer: Initializer):
        self.weights = initializer.get(self.output_size, self.input_size)
        self.bias = initializer.get(self.output_size, 1)

    def forward(self, input: Array) -> Array:
        self.input = input
        return np.dot(self.weights, input) + self.bias
    
    def backward(self, output_gradient: Array) -> BackwardState:
        return BackwardState(
            np.dot(self.weights.T, output_gradient),
            (np.dot(output_gradient, self.input.T), output_gradient)
        )
    
    def input_shape(self) -> Shape:
        return (self.input_size, 1)
    
    def output_shape(self) -> Shape:
        return (self.output_size, 1)

    def trainable(self) -> bool:
        return True

    def parameters(self) -> tuple[Array, ...]:
        return (self.weights, self.bias)