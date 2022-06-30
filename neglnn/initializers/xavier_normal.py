import numpy as np
from neglnn.initializers.initializer import Initializer
from neglnn.utils.types import Array

class XavierNormal(Initializer):
    def get(self, *shape: int) -> Array:
        input_neurons = np.prod(self.state.current_layer_input_shape)
        output_neurons = np.prod(self.state.current_layer_output_shape)
        return np.random.randn(*shape) * np.sqrt(2 / (input_neurons + output_neurons))