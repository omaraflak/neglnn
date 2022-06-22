import numpy as np
from neglnn.layers.layer import Layer, BackwardState
from neglnn.utils.types import Array

class Softmax(Layer):
    def forward(self, input: Array) -> Array:
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp)
        return self.output

    def backward(self, output_gradient: Array) -> BackwardState:
        n = np.size(self.output)
        return BackwardState(np.dot((np.identity(n) - np.transpose(self.output)) * self.output, output_gradient))