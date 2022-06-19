from dataclasses import dataclass, field
from neglnn.network.state import Stateful
from neglnn.initializers.initializer import Initializer
from neglnn.utils.types import Array, Shape
from neglnn.utils.identifiable import Identifiable

@dataclass
class BackwardState:
    input_gradient: Array
    parameter_gradients: tuple[Array, ...] = field(default_factory=tuple)

class Layer(Stateful, Identifiable):
    def initialize(self, initializer: Initializer):
        raise NotImplementedError

    def forward(self, input: Array) -> Array:
        raise NotImplementedError

    def backward(self, output_gradient: Array) -> BackwardState:
        raise NotImplementedError

    def input_shape(self) -> Shape:
        raise NotImplementedError

    def output_shape(self) -> Shape:
        raise NotImplementedError

    def parameters(self) -> tuple[Array, ...]:
        raise NotImplementedError

    def trainable(self) -> bool:
        raise NotImplementedError
    
    def parameters_count(self) -> int:
        return len(self.parameters())