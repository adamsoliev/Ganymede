#!/usr/bin/python3

import numpy as np
from tensor import Tensor

""" Layer - 'nout' number of neurons. Each neuron has 'nin' weights and a bias """
class Linear():
    def __init__(self, nin: int, nout: int) -> None:
        self.w = Tensor(np.random.uniform(-1, 1, (nout, nin)))
        self.b = Tensor(np.random.uniform(-1, 1, (nout)))
    
    def __call__(self, x: 'Tensor') -> 'Tensor':
        return x.matmul(self.w.T()) + self.b

    def parameters(self) -> list['Tensor']:
        return [self.w, self.b]
