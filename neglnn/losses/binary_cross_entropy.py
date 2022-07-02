import numpy as np
from neglnn.losses.loss import Loss
from neglnn.utils.types import Array, Float

class BinaryCrossEntropy(Loss):
    def call(self, true: Array, pred: Array) -> Float:
        return -true * np.log(pred) - (1 - true) * np.log(1 - pred)

    def prime(self, true: Array, pred: Array) -> Array:
        return -true / pred + (1 - true) / (1 - pred)