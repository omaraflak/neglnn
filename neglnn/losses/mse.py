import numpy as np
from neglnn.losses.loss import Loss
from neglnn.utils.types import Array, Float

class MSE(Loss):
    def call(self, true: Array, pred: Array) -> Float:
        return np.mean(np.power(true - pred, 2))

    def prime(self, true: Array, pred: Array) -> Array:
        return 2 * (pred - true) / pred.size