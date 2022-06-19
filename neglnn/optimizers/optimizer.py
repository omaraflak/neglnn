from neglnn.network.state import Stateful
from neglnn.utils.types import Array

class Optimizer(Stateful):
    def record(self, parameters: tuple[Array, ...], gradients: tuple[Array, ...]):
        raise NotImplementedError

    def update(self):
        raise NotImplementedError

    def should_update(self) -> bool:
        raise NotImplementedError