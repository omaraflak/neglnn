import numpy as np
from neglnn.losses.loss import Loss
from neglnn.utils.types import Array, Float

class CrossEntropy(Loss):
    def call(self, true: Array, pred: Array) -> Float:
        return -np.sum(true * np.log(pred))

    def prime(self, true: Array, pred: Array) -> Array:
        return -true / pred