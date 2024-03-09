import unittest
import torch

from tensor import Tensor

class TestTensorSum(unittest.TestCase):
    def test_simple_add(self):
        t1 = Tensor([1., 2., 3.])
        t2 = Tensor([4., 5., 6.])
        t3 = t1 + t2

        assert t3.data.tolist() == [5., 7., 9.]

        t3.backward()

        assert t1.grad.data.tolist() == [1., 1., 1.]
        assert t2.grad.data.tolist() == [1., 1., 1.]
        assert t3.grad.data.tolist() == [1., 1., 1.]
    
    # Broadcasting
    #   1. assume missing dimentions have size one
    #   2. treat dimentions with size one expandable
    def test_broadcast_add(self):
        t1 = Tensor([[3., 4., 5.], [4., 3., 6.]])
        t2 = Tensor([9., 4., 1.])
        t3 = t1 + t2

        assert t3.data.tolist() == [[12.,  8.,  6.], [13.,  7.,  7.]]

        t3.backward()

        assert t1.grad.data.tolist() == [[1., 1., 1.], [1., 1., 1.]]
        assert t2.grad.data.tolist() == [2., 2., 2.]
        assert t3.grad.data.tolist() == [[1., 1., 1.], [1., 1., 1.]]
    
    def test_pytorch_add(self):
        # pytorch
        a = torch.tensor([[0.2606, 0.0398, 0.2312], [0.4034, 0.8265, 0.7248]], requires_grad=True)
        b = torch.tensor(  
            [[0.2026, 0.4692, 0.6961, 0.0221],
            [0.7270, 0.7451, 0.8819, 0.2733],
            [0.8547, 0.2478, 0.0153, 0.8785]], requires_grad=True)
        c = a.matmul(b)
        c.retain_grad()
        d = torch.tensor(   
            [[0.0315, 0.0230, 0.0625, 0.9245],
            [0.6002, 0.0274, 0.2519, 0.3179]], requires_grad=True)
        e = c + d
        e.retain_grad()
        f = e.sum()
        f.retain_grad()

        f.backward() 

        # our torch
        ta = Tensor([[0.2606, 0.0398, 0.2312], [0.4034, 0.8265, 0.7248]])
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

        tf.backward() 

        assert round(f.item(), 5) == round(tf.item(), 5)
        for ta_, a_ in zip(ta.grad.flatten(), a.grad.flatten()):
            assert round(a_.item(), 5) == round(ta_, 5)
        for tb_, b_ in zip(tb.grad.flatten(), b.grad.flatten()):
            assert round(b_.item(), 5) == round(tb_, 5)
        for tc_, c_ in zip(tc.grad.flatten(), c.grad.flatten()):
            assert round(c_.item(), 5) == round(tc_, 5)
        for td_, d_ in zip(td.grad.flatten(), d.grad.flatten()):
            assert round(d_.item(), 5) == round(td_, 5)
        for te_, e_ in zip(te.grad.flatten(), e.grad.flatten()):
            assert round(e_.item(), 5) == round(te_, 5)
