import numpy as np
from neglnn.optimizers.optimizer import Optimizer, Update
from neglnn.utils.types import Float, Shape

class Momentum(Optimizer):
    def __init__(self, learning_rate: Float = 0.01, mu: Float = 0.95):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu

    def record(self, update: Update):
        self.update = update

    def optimize(self):
        self.v = self.mu * self.v + self.learning_rate * self.update.gradient
        self.update.parameter -= self.v

    def should_optimize(self) -> bool:
        return True

    def on_target_shape(self, target_shape: Shape):
        super().on_target_shape(target_shape)
        self.v = np.zeros(target_shape)