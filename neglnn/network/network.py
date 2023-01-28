from dataclasses import dataclass
from typing import Optional, Callable
from neglnn.initializers.initializer import Initializer
from neglnn.layers.layer import Layer
from neglnn.losses.loss import Loss
from neglnn.network.state import State
from neglnn.optimizers.optimizer import Optimizer
from neglnn.utils.types import Array

@dataclass
class Block:
    layer: Layer
    initializer: Optional[Initializer] = None
    provider: Optional[Callable[[], Optimizer]] = None

class Network:
    def __init__(self, network: list[Block]):
        self.network = network

    def fit(
        self,
        x_train: list[Array],
        y_train: list[Array],
        loss: Loss,
        epochs: int,
        verbose: bool = True,
        callback: Optional[Callable[[State], None]] = None 
    ):
        state = self._initialize()
        state.epochs = epochs
        state.training_samples = len(x_train)

        # training loop
        for i in range(epochs):
            state.current_epoch = i
            cost = 0

            # go through all training samples
            for x, y in zip(x_train, y_train):
                # forward propagation
                output = x
                for index, block in enumerate(self.network):
                    state.current_layer = index
                    output = block.layer.forward(output)
                
                # error for display purpose
                cost += loss.call(y, output)

                # backward propagation
                output_gradient = loss.prime(y, output)
                for index, block in enumerate(reversed(self.network)):
                    state.current_layer = index
                    output_state = block.layer.backward(output_gradient)
                    output_gradient = output_state.input_gradient
                    if block.layer.trainable:
                        block.layer.optimize(output_state.parameter_gradients)

            cost /= len(x_train)
            state.cost = cost

            if verbose:
                print(f'#{i + 1}/{epochs}\t cost={cost:2f}')
            
            if callback:
                callback(state)

    def run(self, x: Array) -> Array:
        for block in self.network:
            x = block.layer.forward(x)
        return x

    def run_all(self, x_list: list[Array]) -> list[Array]:
        return [self.run(x) for x in x_list]

    def _initialize(self) -> State:
        state = State(layers=[block.layer for block in self.network])
        
        # provide state to layers
        for block in self.network:
            block.layer.on_state(state)

        # provide state to initializers
        for block in self.network:
            if block.layer.trainable:
                state.current_layer = block.layer
                block.initializer.on_state(state)

        # initialize layers parameters
        for block in self.network:
            if block.layer.trainable:
                block.layer.on_initializer(block.initializer)

        # initialize layers optimizers
        for block in self.network:
            if block.layer.trainable:
                block.layer.on_optimizer(block.provider)

        return state
    
    def __getitem__(self, subscript) -> 'Network':
        return Network(self.network[subscript])