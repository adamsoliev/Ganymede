import random
import numpy as np
from tensor import Tensor

# class Module:
#     def zero_grad(self) -> None:
#         for p in self.parameters():
#             p.grad = np.array(0.0)
    
#     def parameters(self) -> list['Tensor']:
#         return []

class Linear():
    def __init__(self, nin, nout) -> None:
        self.w = Tensor(np.random.uniform(-1, 1, (nout, nin)))
        self.b = Tensor(np.random.uniform(-1, 1, (nout)))
    
    def __call__(self, x: 'Tensor') -> 'Tensor':
        return x.matmul(self.w) + self.b

    def parameters(self) -> list['Tensor']:
        return [self.w, self.b]
    

