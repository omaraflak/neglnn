from neglnn.optimizers.optimizer import Optimizer, Update
from neglnn.utils.types import Float, Shape

class SGD(Optimizer):
    def __init__(self, learning_rate: Float = 0.01):
        super().__init__()
        self.learning_rate = learning_rate
        self.data: Update = None

    def record(self, update: Update):
        self.data = update

    def update(self):
        self.data.parameter -= self.learning_rate * self.data.gradient

    def should_update(self) -> bool:
        return True
    