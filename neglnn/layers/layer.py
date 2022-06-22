from dataclasses import dataclass, field
from typing import Callable
from neglnn.network.state import Stateful
from neglnn.initializers.initializer import Initializer
from neglnn.optimizers.optimizer import Optimizer, Update
from neglnn.utils.types import Array, Shape
from neglnn.utils.identifiable import Identifiable

@dataclass
class BackwardState:
    input_gradient: Array
    parameter_gradients: tuple[Array, ...] = field(default_factory=tuple)

class Layer(Stateful, Identifiable):
    def on_initializer(self, initializer: Initializer):
        raise NotImplementedError
    
    def on_optimizer(self, provider: Callable[[], Optimizer]):
        self.optimizers: list[Optimizer] = []
        for parameter in self.parameters():
            optimizer = provider()
            optimizer.on_state(self.state)
            optimizer.on_target_shape(parameter.shape)
            self.optimizers.append(optimizer)

    def forward(self, input: Array) -> Array:
        raise NotImplementedError

    def backward(self, output_gradient: Array) -> BackwardState:
        raise NotImplementedError

    def input_shape(self) -> Shape:
        raise NotImplementedError

    def output_shape(self) -> Shape:
        raise NotImplementedError

    def trainable(self) -> bool:
        return False

    def parameters(self) -> tuple[Array, ...]:
        raise NotImplementedError
    
    def optimize(self, gradients: tuple[Array, ...]):
        for optimizer, parameter, gradient in zip(
            self.optimizers,
            self.parameters(),
            gradients
        ):
            optimizer.record(Update(parameter, gradient))
            if optimizer.should_optimize():
                optimizer.optimize()