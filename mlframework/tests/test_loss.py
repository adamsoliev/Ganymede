import unittest
import torch
import numpy as np
from tensor import Tensor
from loss import mse_loss, binary_cross_entropy
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

    def test_binary_cross_entropy(self):
        np.random.seed(42)
        y_pred = np.random.rand(100)
        y_true = np.random.randint(0, 2, size=100)

        y_pred_tensor = torch.from_numpy(y_pred).float()
        y_true_tensor = torch.from_numpy(y_true).float()

        y_pred_Tensor = Tensor(y_pred)
        y_true_Tensor = Tensor(y_true)
        loss_custom = binary_cross_entropy(y_pred_Tensor, y_true_Tensor)

        loss_pytorch = torch.nn.functional.binary_cross_entropy(y_pred_tensor, y_true_tensor)

        tolerance = 1e-5
        assert np.allclose(loss_custom.item(), loss_pytorch.item(), atol=tolerance)
    

    def test_bce_loss_backprop(self):
        input = torch.randn(3, 2, requires_grad=True)
        target = torch.rand(3, 2, requires_grad=True)

        m = torch.nn.Sigmoid()
        loss_fn = torch.nn.BCELoss()
        output = loss_fn(m(input), target)
        output.backward()

        # assert isinstance(target.detach().numpy(), np.ndarray)
        # print("$$$ type: ", type(target.detach().numpy()))

        input_tensor = Tensor(input.detach().numpy())
        target_tensor = Tensor(target.detach().numpy())

        tsig = sigmoid(input_tensor)
        toutput = binary_cross_entropy(tsig, target_tensor)
        toutput.backward()

        assert round(output.item(), 5) == round(toutput.item(), 5)
        # for w, tw in zip(W.grad.flatten(), tW.grad.flatten()):
        #     assert round(w.item(), 5) == round(tw.item(), 5)
        # for w, tw in zip(b.grad.flatten(), tb.grad.flatten()):
        #     assert round(w.item(), 5) == round(tw.item(), 5)