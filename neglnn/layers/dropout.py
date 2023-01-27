import numpy as np
from neglnn.layers.layer import Layer, BackwardState
from neglnn.utils.types import Array, Float

class Dropout(Layer):
    def __init__(self, probability: Float = 0.3, training: bool = True):
        super().__init__(trainable=False)
        self.probability = probability
        self.training = training

    def forward(self, input: Array) -> Array:
        if self.training:
            return input
        self.kept = np.random.rand(*input.shape) > self.probability
        return self.kept * input

    def backward(self, output_gradient: Array) -> BackwardState:
        return BackwardState(self.kept * output_gradient)