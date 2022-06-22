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
        self.weights = initializer.get(self.input_size, self.output_size)
        self.bias = initializer.get(1, self.output_size)

    def forward(self, input: Array) -> Array:
        self.input = input
        return np.dot(input, self.weights) + self.bias
    
    def backward(self, output_gradient: Array) -> BackwardState:
        return BackwardState(
            np.dot(output_gradient, self.weights.T),
            (np.dot(self.input.T, output_gradient), output_gradient)
        )
    
    def input_shape(self) -> Shape:
        return (1, self.input_size)
    
    def output_shape(self) -> Shape:
        return (1, self.output_size)

    def trainable(self) -> bool:
        return True

    def parameters(self) -> tuple[Array, ...]:
        return (self.weights, self.bias)