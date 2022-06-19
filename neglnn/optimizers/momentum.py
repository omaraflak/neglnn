import numpy as np
from neglnn.optimizers.optimizer import Optimizer, Update
from neglnn.utils.types import Float, Array, Shape

class Momentum(Optimizer):
    def __init__(self, learning_rate: Float = 0.01, mu: Float = 0.95):
        super().__init__()
        self.learning_rate = learning_rate
        self.mu = mu
        self.v: Array = None
        self.data: Update = None

    def record(self, update: Update):
        self.data = update

    def update(self):
        self.v = self.mu * self.v + self.learning_rate * self.data.gradient
        self.data.parameter -= self.v

    def should_update(self) -> bool:
        return True
    
    def on_target_shape(self, target_shape: Shape):
        super().on_target_shape(target_shape)
        self.v = np.zeros(target_shape)