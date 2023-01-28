import numpy as np
from neglnn.initializers.initializer import Initializer
from neglnn.utils.types import Array

class HeUniform(Initializer):
    def get(self, *shape: int) -> Array:
        input_neurons = np.prod(self.state.current_layer_input_shape)
        limit = np.sqrt(6 / input_neurons)
        return np.random.uniform(-limit, limit, shape)