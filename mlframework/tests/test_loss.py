import unittest
import torch

from tensor import Tensor
from loss import mse_loss

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

