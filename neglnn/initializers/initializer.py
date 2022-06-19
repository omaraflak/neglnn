from neglnn.network.state import Stateful
from neglnn.utils.types import Array

class Initializer(Stateful):
    def get(self, *shape: int) -> Array:
        raise NotImplementedError