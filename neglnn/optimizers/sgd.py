from dataclasses import dataclass
from neglnn.optimizers.optimizer import Optimizer
from neglnn.utils.types import Array, Float

@dataclass
class Update:
    parameter: Array
    gradient: Array

class SGD(Optimizer):
    def __init__(self, learning_rate: Float):
        super().__init__()
        self.learning_rate = learning_rate
        self.updates: list[Update] = []

    def record(self, parameters: tuple[Array, ...], gradients: tuple[Array, ...]):
        self.updates.extend(Update(p, g) for p, g in zip(parameters, gradients))

    def update(self):
        for update in self.updates:
            update.parameter -= self.learning_rate * update.gradient
        self.updates.clear()

    def should_update(self) -> bool:
        return True