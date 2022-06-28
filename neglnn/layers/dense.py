import numpy as np
from neglnn.layers.layer import Layer, BackwardState
from neglnn.initializers.initializer import Initializer
from neglnn.utils.types import Array

class Dense(Layer):
    def __init__(self, input_units: int, output_units: int):
        super().__init__((input_units, 1), (output_units, 1), trainable=True)
        self.input_units = input_units
        self.output_units = output_units
    
    def on_initializer(self, initializer: Initializer):
        self.weights = initializer.get(self.output_units, self.input_units)
        self.bias = initializer.get(self.output_units, 1)

    def forward(self, input: Array) -> Array:
        self.input = input
        return np.dot(self.weights, input) + self.bias
    
    def backward(self, output_gradient: Array) -> BackwardState:
        return BackwardState(
            np.dot(self.weights.T, output_gradient),
            (np.dot(output_gradient, self.input.T), output_gradient)
        )

    def parameters(self) -> tuple[Array, ...]:
        return (self.weights, self.bias)