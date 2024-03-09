import unittest
import torch

from tensor import Tensor

class TestTensor(unittest.TestCase):
    def test_simple_neg(self):
        na = [1., 2., 3.]
        nb = [4., 5., 6.]

        p1 = torch.tensor(na, requires_grad=True)
        p2 = torch.tensor(nb, requires_grad=True)
        p3 = p1 + p2; p3.retain_grad()
        p4 = -p3; p4.retain_grad()
        p5 = p4.sum(); p5.retain_grad()

        p5.backward()

        t1 = Tensor(na)
        t2 = Tensor(nb)
        t3 = t1 + t2
        t4 = -t3
        t5 = t4.sum()
        t5.backward()

        assert round(p5.item(), 5) == round(t5.item(), 5)
        for ta_, a_ in zip(t1.grad.flatten(), p1.grad.flatten()):
            assert round(a_.item(), 5) == round(ta_, 5)
        for tb_, b_ in zip(t2.grad.flatten(), p2.grad.flatten()):
            assert round(b_.item(), 5) == round(tb_, 5)
        for tc_, c_ in zip(t3.grad.flatten(), p3.grad.flatten()):
            assert round(c_.item(), 5) == round(tc_, 5)
        for td_, d_ in zip(t4.grad.flatten(), p4.grad.flatten()):
            assert round(d_.item(), 5) == round(td_, 5)
        for te_, e_ in zip(t5.grad.flatten(), p5.grad.flatten()):
            assert round(e_.item(), 5) == round(te_, 5)
        

    def test_add(self):
        def _helper(na: list, nb: list):
            p1 = torch.tensor(na, requires_grad=True)
            p2 = torch.tensor(nb, requires_grad=True)
            p3 = p1 + p2; p3.retain_grad()
            p4 = p3.sum(); p4.retain_grad()
            p4.backward()

            t1 = Tensor(na)
            t2 = Tensor(nb)
            t3 = t1 + t2
            t4 = t3.sum()
            t4.backward()

            assert round(p4.item(), 5) == round(t4.item(), 5)
            for ta_, a_ in zip(t1.grad.flatten(), p1.grad.flatten()):
                assert round(a_.item(), 5) == round(ta_, 5)
            for tb_, b_ in zip(t2.grad.flatten(), p2.grad.flatten()):
                assert round(b_.item(), 5) == round(tb_, 5)
            for tc_, c_ in zip(t3.grad.flatten(), p3.grad.flatten()):
                assert round(c_.item(), 5) == round(tc_, 5)
            for td_, d_ in zip(t4.grad.flatten(), p4.grad.flatten()):
                assert round(d_.item(), 5) == round(td_, 5)

        na = [1., 2., 3.]; nb = [4., 5., 6.]
        _helper(na, nb)
        na = [[1., 2., 3.]]; nb = [4., 5., 6.]
        _helper(na, nb)
        na = [1., 2., 3.]; nb = [[4., 5., 6.]]
        _helper(na, nb)
        # broadcasting 
        #   1. assume missing dimentions have size one 
        #   2. treat dimentions with size one expandable
        na = [[3., 4., 5.], [4., 3., 6.]]; nb = [9., 4., 1.]    # (2,3) <-> (3,)
        _helper(na, nb)
        na = [[3., 4., 5.], [4., 3., 6.]]; nb = [[9., 4., 1.]]  # (2,3) <-> (1,3)
        _helper(na, nb)
    

    def test_mul(self):
        def _helper(na: list, nb: list):
            p1 = torch.tensor(na, requires_grad=True)
            p2 = torch.tensor(nb, requires_grad=True)
            p3 = p1 * p2; p3.retain_grad()
            p4 = p3.sum(); p4.retain_grad()
            p4.backward()

            t1 = Tensor(na)
            t2 = Tensor(nb)
            t3 = t1 * t2
            t4 = t3.sum()
            t4.backward()

            assert round(p4.item(), 5) == round(t4.item(), 5)
            for ta_, a_ in zip(t1.grad.flatten(), p1.grad.flatten()):
                assert round(a_.item(), 5) == round(ta_, 5)
            for tb_, b_ in zip(t2.grad.flatten(), p2.grad.flatten()):
                assert round(b_.item(), 5) == round(tb_, 5)
            for tc_, c_ in zip(t3.grad.flatten(), p3.grad.flatten()):
                assert round(c_.item(), 5) == round(tc_, 5)
            for td_, d_ in zip(t4.grad.flatten(), p4.grad.flatten()):
                assert round(d_.item(), 5) == round(td_, 5)

        na = [1., 2., 3.]; nb = [4., 5., 6.]
        _helper(na, nb)
        na = [[1., 2., 3.]]; nb = [4., 5., 6.]
        _helper(na, nb)
        na = [1., 2., 3.]; nb = [[4., 5., 6.]]
        _helper(na, nb)
        # broadcasting 
        #   1. assume missing dimentions have size one 
        #   2. treat dimentions with size one expandable
        na = [[3., 4., 5.], [4., 3., 6.]]; nb = [9., 4., 1.]    # (2,3) <-> (3,)
        _helper(na, nb)
        na = [[3., 4., 5.], [4., 3., 6.]]; nb = [[9., 4., 1.]]  # (2,3) <-> (1,3)
        _helper(na, nb)
    

    def test_pytorch_compare(self):
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