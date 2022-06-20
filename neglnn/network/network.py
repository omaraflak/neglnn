from dataclasses import dataclass, field
from typing import Optional, Callable
from neglnn.initializers.initializer import Initializer
from neglnn.layers.layer import Layer
from neglnn.losses.loss import Loss
from neglnn.network.state import State
from neglnn.optimizers.optimizer import Optimizer, Update
from neglnn.utils.types import Array

@dataclass
class Block:
    layer: Layer
    initializer: Optional[Initializer] = None
    optimizers: list[Optimizer] = field(default_factory=list)

@dataclass
class BlockBuilder:
    layer: Layer
    initializer: Optional[Initializer] = None
    optimizer_provider: Optional[Callable[[], Optimizer]] = None

    def optimizers(self, n: int) -> list[Optimizer]:
        return [self.optimizer_provider() for _ in range(n)]

    def build(self):
        optimizers: list[Optimizer] = []
        if self.layer.trainable():
            optimizers = self.optimizers(self.layer.parameters_count())
        return Block(self.layer, self.initializer, optimizers)

class Network:
    def __init__(self, network: list[Block]):
        self.network = network

    def fit(
        self,
        x_train: list[Array],
        y_train: list[Array],
        loss: Loss,
        epochs: int,
        verbose: bool = True
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
                    if block.layer.trainable():
                        for optimizer, parameter, gradient in zip(
                            block.optimizers,
                            block.layer.parameters(),
                            output_state.parameter_gradients
                        ):
                            optimizer.record(Update(parameter, gradient))
                            if optimizer.should_optimize():
                                optimizer.optimize()

            cost /= len(x_train)
            state.cost = cost

            if verbose:
                print(f'#{i + 1}/{epochs}\t cost={cost:2f}')

    def predict(self, x: Array) -> Array:
        for block in self.network:
            x = block.layer.forward(x)
        return x

    def predict_all(self, x_list: list[Array]) -> list[Array]:
        return [self.predict(x) for x in x_list]

    def _initialize(self) -> State:
        state = State(layers=[block.layer for block in self.network])
        
        # provide state to layers
        for block in self.network:
            block.layer.on_state(state)

        # provide state to initializers
        for block in self.network:
            if block.layer.trainable():
                block.initializer.on_state(state)

        # provide state to optimizers
        for block in self.network:
            if block.layer.trainable():
                for optimizer in block.optimizers:
                    optimizer.on_state(state)

        # initialize layers parameters
        for block in self.network:
            if block.layer.trainable():
                block.layer.initialize(block.initializer)

        # provide initialized target shapes to optimizers
        for block in self.network:
            if block.layer.trainable():
                for optimizer, parameter in zip(block.optimizers, block.layer.parameters()):
                    optimizer.on_target_shape(parameter.shape)

        return state
    
    def __getitem__(self, subscript) -> 'Network':
        return Network(self.network[subscript])

    @staticmethod
    def create(builders: list[BlockBuilder]) -> 'Network':
        return Network([builder.build() for builder in builders])