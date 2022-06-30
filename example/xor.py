import numpy as np
from neglnn.layers.dense import Dense
from neglnn.activations.tanh import Tanh
from neglnn.losses.mse import MSE
from neglnn.initializers.random_uniform import RandomUniform
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import Network, Block

x_train = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
y_train = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = Network([
    Block(Dense(2, 3), RandomUniform(), lambda: Momentum()),
    Block(Tanh()),
    Block(Dense(3, 1), RandomUniform(), lambda: Momentum()),
    Block(Tanh())
])

network.fit(x_train, y_train, MSE(), 1000)

print(network.run_all(x_train))