from keras.datasets import mnist
from keras.utils import np_utils

from neglnn.layers.dense import Dense
from neglnn.layers.reshape import Reshape
from neglnn.activations.tanh import Tanh
from neglnn.activations.softmax import Softmax
from neglnn.losses.mse import MSE
from neglnn.initializers.xavier_normal import XavierNormal
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import Network, Block

def load_data(limit: int):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)

    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)

    return x_train[:limit], y_train[:limit], x_test, y_test

network = Network([
    Block(Reshape((28, 28), (784, 1))),
    Block(Dense(784, 50), XavierNormal(), lambda: Momentum(0.1)),
    Block(Tanh()),
    Block(Dense(50, 20), XavierNormal(), lambda: Momentum(0.1)),
    Block(Tanh()),
    Block(Dense(20, 10), XavierNormal(), lambda: Momentum(0.1)),
    Block(Softmax())
])

x_train, x_test = load_data(1000)

network.fit(x_train, x_train, MSE(), 50)