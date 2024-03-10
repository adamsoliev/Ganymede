from tensor import Tensor

class SGD:
    def __init__(self, params: list['Tensor'], lr: float = 0.01) -> None:
        self.learning_rate = lr
        self.params = params
    
    def step(self) -> None:
        for param in self.params:
            param.data -= param.grad * self.learning_rate
