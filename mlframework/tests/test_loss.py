import unittest
import torch
import numpy as np
from tensor import Tensor
from loss import mse_loss, bce_loss
from nn import sigmoid

class TestLoss(unittest.TestCase):
    def test_mse_loss(self):
        def _helper(W_, X_, b_, y_):
            W = torch.tensor(W_, requires_grad=True)
            X = torch.tensor(X_)
            b = torch.tensor(b_, requires_grad=True)
            y = torch.tensor(y_)
            y_prime = torch.matmul(W, X) + b
            loss_fn = torch.nn.MSELoss()
            loss = loss_fn(y_prime, y)
            loss.backward()

            tW = Tensor(W_)
            tX = Tensor(X_)
            tb = Tensor(b_)
            ty = Tensor(y_)
            ty_prime_ = tW.matmul(tX) + tb
            loss_ = mse_loss(ty_prime_, ty)
            loss_.backward()

            assert round(loss.item(), 5) == round(loss_.item(), 5)
            for w, tw in zip(W.grad.flatten(), tW.grad.flatten()):
                assert round(w.item(), 5) == round(tw.item(), 5)
            for w, tw in zip(b.grad.flatten(), tb.grad.flatten()):
                assert round(w.item(), 5) == round(tw.item(), 5)

        W = [[1., 2.]]
        X = [[3.], [4.]]
        b = [5.]
        y = [[6.]]
        _helper(W, X, b, y)

    def test_bce_loss(self):
        def _helper(ninput, ntarget):
            pinput = torch.from_numpy(ninput); pinput.requires_grad=True
            ptarget = torch.from_numpy(ninput); ptarget.requires_grad=True
            loss_fn = torch.nn.BCELoss()
            psig = torch.sigmoid(pinput); print(psig)
            ploss = loss_fn(psig, ptarget)
            ploss.backward()

            tinput = Tensor(ninput)
            ttarget = Tensor(ntarget)

            tsig = sigmoid(tinput); print(tsig)
            tloss = bce_loss(ttarget, tsig)
            tloss.backward()

            assert round(tloss.item(), 5) == round(ploss.item(), 5)
            for w, tw in zip(tinput.grad.flatten(), pinput.grad.flatten()):
                assert round(w.item(), 5) == round(tw.item(), 5)
            for w, tw in zip(ttarget.grad.flatten(), ptarget.grad.flatten()):
                assert round(w.item(), 5) == round(tw.item(), 5)

        input = np.random.rand(3, 2)
        target = np.array([[1., 0.], [1., 1.], [0., 1.]])
        _helper(input, target)
