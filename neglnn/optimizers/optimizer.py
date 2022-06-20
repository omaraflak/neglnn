from dataclasses import dataclass
from typing import Optional
from neglnn.network.state import Stateful
from neglnn.utils.types import Array, Shape

@dataclass
class Update:
    parameter: Array
    gradient: Array

class Optimizer(Stateful):
    def __init__(self):
        self.target_shape: Optional[Shape] = None

    def record(self, update: Update):
        raise NotImplementedError

    def optimize(self):
        raise NotImplementedError

    def should_optimize(self) -> bool:
        raise NotImplementedError

    def on_target_shape(self, target_shape: Shape):
        self.target_shape = target_shape