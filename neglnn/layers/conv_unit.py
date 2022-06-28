import numpy as np
from scipy import signal
from neglnn.layers.layer import Layer, BackwardState
from neglnn.initializers.initializer import Initializer
from neglnn.utils.types import Array, Shape3

class ConvUnit(Layer):
    def __init__(self, input_shape: Shape3, kernel_size: int):
        input_depth, input_height, input_width = input_shape
        output_shape = (
            input_height - kernel_size + 1,
            input_width - kernel_size + 1
        )
        super().__init__(input_shape, output_shape, trainable=True)
        self.kernels_shape = (input_depth, kernel_size, kernel_size)
        self.bias_shape = output_shape
    
    def on_initializer(self, initializer: Initializer):
        self.kernels = initializer.get(*self.kernels_shape)
        self.bias = initializer.get(*self.bias_shape)

    def forward(self, input: Array) -> Array:
        self.input = input
        return self.bias + np.sum([
            signal.correlate2d(image, kernel, 'valid')
            for image, kernel in zip(input, self.kernels)
        ], axis=0)
    
    def backward(self, output_gradient: Array) -> BackwardState:
        input_gradient = np.array([
            signal.convolve2d(output_gradient, kernel, 'full')
            for kernel in self.kernels
        ])

        kernels_gradient = np.array([
            signal.correlate2d(image, output_gradient, 'valid')
            for image in self.input
        ])

        return BackwardState(input_gradient, [kernels_gradient, output_gradient])

    def parameters(self) -> tuple[Array, ...]:
        return (self.kernels, self.bias)