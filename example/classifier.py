import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils

from neglnn.layers.dense import Dense
from neglnn.layers.flatten import Flatten
from neglnn.activations.tanh import Tanh
from neglnn.activations.softmax import Softmax
from neglnn.losses.mse import MSE
from neglnn.initializers.xavier import Xavier
from neglnn.optimizers.sgd import SGD
from neglnn.network.network import Network, Block

def load_data(limit: int):
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train = x_train.astype('float32')
    x_train /= 255
    y_train = np_utils.to_categorical(y_train)
    y_train = y_train.reshape(y_train.shape[0], 10, 1)

    x_test = x_test.astype('float32')
    x_test /= 255
    y_test = np_utils.to_categorical(y_test)
    y_test = y_test.reshape(y_test.shape[0], 10, 1)

    return x_train[:limit], y_train[:limit], x_test, y_test

network = Network([
    Block(Flatten((28, 28))),
    Block(Dense(784, 50), Xavier(), lambda: SGD(0.1)),
    Block(Tanh()),
    Block(Dense(50, 20), Xavier(), lambda: SGD(0.1)),
    Block(Tanh()),
    Block(Dense(20, 10), Xavier(), lambda: SGD(0.1)),
    Block(Softmax())
])

x_train, y_train, x_test, y_test = load_data(1000)

network.fit(x_train, y_train, MSE(), 50)

for x, y in zip(x_test[:20], y_test[:20]):
    prob = network.run(x)
    pred = np.argmax(prob)
    true = np.argmax(y)
    print(f'pred={pred} at {prob[pred][0]:.2f}, true={true}')