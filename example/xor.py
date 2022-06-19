import numpy as np
from neglnn.layers.dense import Dense
from neglnn.activations.tanh import Tanh
from neglnn.losses.mse import MSE
from neglnn.initializers.normal import Normal
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import create, fit, predict

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 1, 2))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = create([
    (Dense(2, 3), Normal(), lambda: Momentum()),
    (Tanh(), None, None),
    (Dense(3, 1), Normal(), lambda: Momentum()),
    (Tanh(), None, None)
])

fit(network, X, Y, MSE(), 1000)

for x in X:
    print(predict(network, x))
