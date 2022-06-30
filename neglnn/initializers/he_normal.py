import numpy as np
from neglnn.initializers.initializer import Initializer
from neglnn.utils.types import Array

class HeNormal(Initializer):
    def get(self, *shape: int) -> Array:
        input_neurons = np.prod(self.state.current_layer_input_shape)
        standard_deviation = np.sqrt(2 / input_neurons)
        return np.random.normal(0, standard_deviation, shape)