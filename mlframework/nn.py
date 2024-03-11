import random
import numpy as np
from tensor import Tensor

class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = np.array(0.0)
    
    def parameters(self) -> list['Tensor']:
        return []
