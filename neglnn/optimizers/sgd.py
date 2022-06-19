from typing import Optional
from neglnn.optimizers.optimizer import Optimizer
from neglnn.utils.types import Array, Float

class SGD(Optimizer):
    def __init__(self, learning_rate: Float):
        self.learning_rate = learning_rate
        self.parameters: Optional[tuple[Array, ...]] = None
        self.gradients: Optional[tuple[Array, ...]] = None

    def record(self, parameters: tuple[Array, ...], gradients: tuple[Array, ...]):
        self.parameters = parameters
        self.gradients = gradients

    def update(self):
        for param, gradient in zip(self.parameters, self.gradients):
            param -= self.learning_rate * gradient
    
    def should_update(self) -> bool:
        return True