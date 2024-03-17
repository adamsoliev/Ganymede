#!/usr/bin/python3

import torch
import numpy as np
from tensor import Tensor

class Linear():
    def __init__(self, nin: int, nout: int) -> None:
        # self.w = Tensor(np.random.uniform(-1, 1, (nout, nin)))
        # self.b = Tensor(np.random.uniform(-1, 1, (nout)))
        self.w = Tensor([[-0.5773, -0.0292],
                        [ 0.4392, -0.6857],
                        [-0.6855,  0.1465]])
        self.b = Tensor([-0.3793,  0.5138,  0.6871])
    
    def __call__(self, x: 'Tensor') -> 'Tensor':
        return x.matmul(self.w.T()) + self.b

    def parameters(self) -> list['Tensor']:
        return [self.w, self.b]
    

def main() -> None:
    torch.manual_seed(13)

    m0 = torch.nn.Linear(2, 3)
    m1 = Linear(2,3)

    input0 = torch.randn(5, 2)
    input1 = Tensor(input0.numpy())

    output0 = m0(input0)
    output1 = m1(input1)
    print(output0)
    print(output1)


if __name__ == '__main__':
    main()