from typing import Optional
from dataclasses import dataclass
from neglnn.utils.types import Float

@dataclass
class State:
    max_iterations: int
    training_samples: int
    layers_count: int
    current_iteration: int = 0
    current_layer: int = 0
    cost: Float = 0

class Stateful:
    def __init__(self):
        self.state: Optional[State] = None
    
    def provide_state(self, state: State):
        self.state = state