import numpy as np
from neglnn.layers.dense import Dense
from neglnn.activations.tanh import Tanh
from neglnn.losses.mse import MSE
from neglnn.initializers.normal import Normal
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import Network, BlockBuilder

x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 1, 2))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = Network.create([
    BlockBuilder(Dense(2, 3), Normal(), lambda: Momentum()),
    BlockBuilder(Tanh()),
    BlockBuilder(Dense(3, 1), Normal(), lambda: Momentum()),
    BlockBuilder(Tanh())
])

network.fit(x_train, y_train, MSE(), 1000)

print(network.run_all(x_train))