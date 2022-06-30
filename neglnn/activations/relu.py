from neglnn.activations.leaky_relu import LeakyRelu

class Relu(LeakyRelu):
    def __init__(self):
        super().__init__(p=0)