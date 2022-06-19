import numpy as np
from neglnn.layers.dense import Dense
from neglnn.activations.tanh import Tanh
from neglnn.losses.mse import MSE
from neglnn.initializers.normal import Normal
from neglnn.optimizers.sgd import SGD
from neglnn.network.network import fit, initialize, predict

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 1, 2))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    (Dense(2, 3), Normal(), SGD(0.01)),
    (Tanh(), None, None),
    (Dense(3, 1), Normal(), SGD(0.01)),
    (Tanh(), None, None)
]

initialize(network)

fit(network, X, Y, MSE(), 10000)
print(predict(network, X[0]))
print(predict(network, X[1]))
print(predict(network, X[2]))
print(predict(network, X[3]))
