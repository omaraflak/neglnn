from neglnn.layers.layer import Layer, BackwardState
from neglnn.utils.types import Array

class Activation(Layer):
    def call(self, x: Array) -> Array:
        raise NotImplementedError
    
    def prime(self, x: Array) -> Array:
        raise NotImplementedError

    def forward(self, input: Array) -> Array:
        self.input = input
        return self.call(input)
    
    def backward(self, output_gradient: Array) -> BackwardState:
        return BackwardState(output_gradient * self.prime(self.input))