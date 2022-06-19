from typing import Optional, Callable
from neglnn.initializers.initializer import Initializer
from neglnn.layers.layer import Layer
from neglnn.losses.loss import Loss
from neglnn.network.state import State
from neglnn.optimizers.optimizer import Optimizer, Update
from neglnn.utils.types import Array

NetworkLayerBuilder = tuple[
    Layer,
    Optional[Initializer],
    Optional[Callable[[], Optimizer]]
]

NetworkLayer = tuple[
    Layer,
    Optional[Initializer],
    list[Optimizer]
]

def create(builder: list[NetworkLayerBuilder]) -> NetworkLayer:
    network: list[NetworkLayer] = []
    for layer, initializer, provider in builder:
        if layer.trainable():
            optimizers = [
                provider()
                for _ in range(layer.parameters_count())
            ]
            network.append((layer, initializer, optimizers))
        else:
            network.append((layer, None, None))
    return network

def initialize(network: list[NetworkLayer]):
    # initialize layers
    for layer, initializer, _ in network:
        if layer.trainable():
            layer.initialize(initializer)

def predict(network: list[NetworkLayer], x: Array) -> Array:
    for layer, _, _ in network:
        x = layer.forward(x)
    return x

def fit(
    network: list[NetworkLayer],
    x_train: list[Array],
    y_train: list[Array],
    loss: Loss,
    iterations: int,
    verbose: bool = True
):
    state = State(
        [layer for layer, _, _ in network],
        iterations,
        len(x_train),
        len(network)
    )

    # provide state to network components
    for layer, initializer, optimizers in network:
        layer.on_state(state)
        if layer.trainable():
            initializer.on_state(state)
            for optimizer in optimizers:
                optimizer.on_state(state)

    loss.on_state(state)

    # initialize layers
    initialize(network)

    # provide initialized target shapes to optimizers
    for layer, _, optimizers in network:
        if layer.trainable():
            for optimizer, parameter in zip(optimizers, layer.parameters()):
                optimizer.on_target_shape(parameter.shape)

    # training loop
    for i in range(iterations):
        state.current_iteration = i
        cost = 0

        # go through all training samples
        for x, y in zip(x_train, y_train):
            # forward propagation
            output = x
            for index, (layer, _, _) in enumerate(network):
                state.current_layer = index
                output = layer.forward(output)
            
            # error for display purpose
            cost += loss.call(y, output)

            # backward propagation
            output_gradient = loss.prime(y, output)
            for index, (layer, _, optimizers) in enumerate(reversed(network)):
                state.current_iteration = index
                output_state = layer.backward(output_gradient)
                output_gradient = output_state.input_gradient
                if layer.trainable():
                    for optimizer, parameter, gradient in zip(
                        optimizers,
                        layer.parameters(),
                        output_state.parameter_gradients
                    ):
                        optimizer.record(Update(parameter, gradient))
                        if optimizer.should_update():
                            optimizer.update()

        cost /= len(x_train)
        state.cost = cost

        if verbose:
            print(f'#{i + 1}/{iterations}\t cost={cost:2f}')