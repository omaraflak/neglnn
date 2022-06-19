from neglnn.layers.layer import Layer, BackwardState
from neglnn.utils.types import Array

class Activation(Layer):
    def activation(self, x: Array) -> Array:
        raise NotImplementedError
    
    def activation_prime(self, x: Array) -> Array:
        raise NotImplementedError

    def forward(self, input: Array) -> Array:
        self.input = input
        return self.activation(input)
    
    def backward(self, output_gradient: Array) -> BackwardState:
        return BackwardState(output_gradient * self.activation_prime(self.input), None)
    
    def trainable(self) -> bool:
        return False