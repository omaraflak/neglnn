from typing import Optional
from dataclasses import dataclass
from neglnn.utils.types import Float, Shape
import neglnn.layers.layer

@dataclass
class State:
    layers: list['neglnn.layers.layer.Layer']
    max_iterations: int
    training_samples: int
    layers_count: int
    current_iteration: int = 0
    current_layer: int = 0
    cost: Float = 0

    def current_layer_input_shape(self) -> Shape:
        return self.layers[self.current_layer].input_shape()

    def current_layer_output_shape(self) -> Shape:
        return self.layers[self.current_layer].output_shape()

class Stateful:
    def __init__(self):
        self.state: Optional[State] = None
    
    def provide_state(self, state: State):
        self.state = state