import numpy as np
from neglnn.layers.layer import Layer, BackwardState
from neglnn.layers.conv_unit import ConvUnit
from neglnn.initializers.initializer import Initializer
from neglnn.utils.types import Array, Shape3

class Conv(Layer):
    def __init__(self, input_shape: Shape3, kernel_size: int, depth: int):
        self.conv_units = [ConvUnit(input_shape, kernel_size) for _ in range(depth)]
        height, width = self.conv_units[0].output_shape
        super().__init__(input_shape, (depth, height, width), trainable=True)
    
    def on_initializer(self, initializer: Initializer):
        for unit in self.conv_units:
            unit.on_initializer(initializer)
        self._parameters = tuple(x for unit in self.conv_units for x in unit.parameters())

    def forward(self, input: Array) -> Array:
        return np.array([unit.forward(input) for unit in self.conv_units])
    
    def backward(self, output_gradient: Array) -> BackwardState:
        back = [unit.backward(grad) for unit, grad in zip(self.conv_units, output_gradient)]
        return BackwardState(
            np.sum([b.input_gradient for b in back], axis=0),
            tuple(grad for b in back for grad in b.parameter_gradients)
        )

    def parameters(self) -> tuple[Array, ...]:
        return self._parameters
