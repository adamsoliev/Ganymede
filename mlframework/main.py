#!/usr/bin/python3

import torch
from torch import nn
import numpy as np

class Tensor:
    def __init__(self, data):
        if type(data) == np.ndarray:
            self.data = data
        else:
            self.data = np.array(data)
    
    def sum(self):
        return self.data.sum()
    
    def matmul(self, other):
        assert type(other) == Tensor
        return Tensor(np.matmul(self.data, other.data))

    def __add__(self, other):
        assert type(other) == Tensor
        return Tensor(self.data + other.data)

    def __repr__(self):
        return f"Tensor: \n{self.data}"

def main():
    # ------------------ PYTORCH ------------------
    torch.set_printoptions(precision=9)

    a = torch.tensor(   # 2x3
        [[0.2606, 0.0398, 0.2312],
         [0.4034, 0.8265, 0.7248]], requires_grad=True)
    b = torch.tensor(   # 3x4
        [[0.2026, 0.4692, 0.6961, 0.0221],
         [0.7270, 0.7451, 0.8819, 0.2733],
         [0.8547, 0.2478, 0.0153, 0.8785]], requires_grad=True)
    c = a.matmul(b)
    c.retain_grad()
    d = torch.tensor(   # 2x4
        [[0.0315, 0.0230, 0.0625, 0.9245],
         [0.6002, 0.0274, 0.2519, 0.3179]], requires_grad=True)
    e = c + d
    e.retain_grad()
    f = e.sum()
    f.retain_grad()

    # f.backward() 

    # ------------------ NUMPY ------------------
    ta = Tensor([[0.2606, 0.0398, 0.2312], [0.4034, 0.8265, 0.7248]])
    # na = np.array([     # 2x3
    #     [0.2606, 0.0398, 0.2312],
    #     [0.4034, 0.8265, 0.7248]])
    # nb = np.array([     # 3x4
    #     [0.2026, 0.4692, 0.6961, 0.0221],
    #     [0.7270, 0.7451, 0.8819, 0.2733],
    #     [0.8547, 0.2478, 0.0153, 0.8785]])
    # nc = np.matmul(na, nb)
    # nd = np.array([      # 2x4
    #     [0.0315, 0.0230, 0.0625, 0.9245],
    #     [0.6002, 0.0274, 0.2519, 0.3179]])
    # ne = nc + nd
    # nf = ne.sum()

    tb = Tensor([
        [0.2026, 0.4692, 0.6961, 0.0221],
        [0.7270, 0.7451, 0.8819, 0.2733],
        [0.8547, 0.2478, 0.0153, 0.8785]])
    tc = ta.matmul(tb)
    td = Tensor([
        [0.0315, 0.0230, 0.0625, 0.9245],
        [0.6002, 0.0274, 0.2519, 0.3179]])
    te = tc + td
    tf = te.sum()

    assert round(f.item(), 5) == round(tf, 5)


    
if __name__ == '__main__':
    main()
