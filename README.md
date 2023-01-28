# NEGLNN

**N**ot **E**fficient but **G**reat to **L**earn **N**eural **N**etwork

# Example

```python
import numpy as np
from neglnn.layers.dense import Dense
from neglnn.activations.tanh import Tanh
from neglnn.losses.mse import MSE
from neglnn.initializers.normal import Normal
from neglnn.optimizers.momentum import Momentum
from neglnn.network.network import Network, BlockBuilder

X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1))
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = Network.create([
    BlockBuilder(Dense(2, 3), Normal(), lambda: Momentum()),
    BlockBuilder(Tanh()),
    BlockBuilder(Dense(3, 1), Normal(), lambda: Momentum()),
    BlockBuilder(Tanh())
])

network.fit(X, Y, MSE(), 1000)

print(network.predict_all(X))
```