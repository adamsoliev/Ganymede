import unittest
import torch
import numpy as np
from tensor import Tensor
from nn import Linear
from pytest import approx

class TestNN(unittest.TestCase):
    def test_linear_forward(self):
        torch.manual_seed(13)

        nin = 2; nout = 3
        nweight = np.random.uniform(-1, 1, (nout, nin))
        nbias = np.random.uniform(-1, 1, (nout))

        # Pytorch linear with fixed weight/bias
        m0 = torch.nn.Linear(nin, nout)
        with torch.no_grad():
            m0.weight.copy_(torch.Tensor(nweight))
            m0.bias.copy_(torch.Tensor(nbias))

        # Our linear with fixed weight/bias
        m1 = Linear(2,3)
        m1.w = Tensor(nweight)
        m1.b = Tensor(nbias)

        input0 = torch.randn(5, 2)
        input1 = Tensor(input0.numpy())

        output0 = m0(input0).detach().numpy().flatten()
        output1 = m1(input1).numpy().flatten()
        for out0, out1 in zip(output0, output1):
            assert out0 == approx(out1)


