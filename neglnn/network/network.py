from typing import Optional
from neglnn.initializers.initializer import Initializer
from neglnn.layers.layer import Layer
from neglnn.losses.loss import Loss
from neglnn.network.state import State
from neglnn.optimizers.optimizer import Optimizer
from neglnn.utils.types import Array

NetworkLayer = tuple[Layer, Optional[Initializer], Optional[Optimizer]]

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
    state = State(iterations, len(x_train), len(network))

    # provide state to network components
    for layer, initializer, optimizer in network:
        layer.provide_state(state)
        if layer.trainable():
            initializer.provide_state(state)
            optimizer.provide_state(state)

    loss.provide_state(state)

    # initialize layers
    initialize(network)

    # training loop
    for i in range(iterations):
        state.current_iteration = i
        cost = 0

        # go through all training samples
        for index, (x, y) in enumerate(zip(x_train, y_train)):
            state.current_layer = index

            # forward propagation
            output = predict(network, x)

            # error for display purpose
            cost += loss.call(y, output)

            # backward propagation
            output_gradient = loss.prime(y, output)
            for layer, _, optimizer in reversed(network):
                output_state = layer.backward(output_gradient)
                output_gradient = output_state.input_gradient
                if layer.trainable():
                    optimizer.record(layer.parameters(), output_state.parameter_gradients)
                    if optimizer.should_update():
                        optimizer.update()

        cost /= len(x_train)
        state.cost = cost

        if verbose:
            print(f'#{i + 1}/{iterations}\t error={cost:2f}')