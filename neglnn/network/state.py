from typing import Optional
from dataclasses import dataclass, field
from neglnn.utils.types import Float, Shape
import neglnn.layers.layer

@dataclass
class State:
    layers: list['neglnn.layers.layer.Layer'] = field(default_factory=list)
    epochs: int = 0
    training_samples: int = 0
    current_epoch: int = 0
    current_layer: int = 0
    cost: Float = 0

    @property
    def current_layer_input_shape(self) -> Shape:
        return self.layers[self.current_layer].input_shape()

    @property
    def current_layer_output_shape(self) -> Shape:
        return self.layers[self.current_layer].output_shape()
    
    @property
    def layers_count(self) -> int:
        return len(self.layers)

class Stateful:
    def __init__(self):
        self.state: Optional[State] = None
    
    def on_state(self, state: State):
        self.state = state