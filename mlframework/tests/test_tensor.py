import unittest

import sys
 
# setting path
sys.path.append('../')

import numpy as np
from tinygrad import Tensor as tinyTensor # type:ignore
from tinygrad import dtypes # type:ignore
from tensor import Tensor

class TestTensor2D(unittest.TestCase):
    def test_bce(self):
        input = np.random.randn(3, 2)
        target = np.random.rand(3, 2)

        a = tinyTensor(input, dtype=dtypes.float32)
        b = tinyTensor(target, dtype=dtypes.float32)
        c = a.sigmoid()
        loss = c.binary_crossentropy(b)

        _a = Tensor(input)
        _b = Tensor(target)
        _c = _a.sigmoid()
        _loss = _c.binary_crossentropy(_b)

        # dot = draw_dot(_loss)
        # dot.view()

        assert np.allclose(c.numpy(), _c.numpy(), atol=1e-6)
        assert np.allclose(loss.numpy(), _loss.numpy(), atol=1e-6)

    def test_add_sub_mul_div(self):
        input = np.random.randn(3, 2)
        target = np.random.rand(3, 2)

        a = tinyTensor(input, dtype=dtypes.float32)
        b = tinyTensor(target, dtype=dtypes.float32)
        c = a + b
        d = a - b
        e = a * b
        f = a / b

        _a = Tensor(input)
        _b = Tensor(target)
        _c = _a + _b
        _d = _a - _b
        _e = _a * _b
        _f = _a / _b

        assert np.allclose(c.numpy(), _c.numpy(), atol=1e-6)
        assert np.allclose(d.numpy(), _d.numpy(), atol=1e-6)
        assert np.allclose(e.numpy(), _e.numpy(), atol=1e-6)
        assert np.allclose(f.numpy(), _f.numpy(), atol=1e-6)

    def test_radd_rsub_rmul_rdiv(self):
        input = np.random.rand(3, 2)

        b = tinyTensor(input, dtype=dtypes.float32)
        c = 1 + b
        d = 1 - b
        e = 1 * b
        f = 1 / b

        _b = Tensor(input)
        _c = 1 + _b
        _d = 1 - _b
        _e = 1 * _b
        _f = 1 / _b

        assert np.allclose(c.numpy(), _c.numpy(), atol=1e-6)
        assert np.allclose(d.numpy(), _d.numpy(), atol=1e-6)
        assert np.allclose(e.numpy(), _e.numpy(), atol=1e-6)
        assert np.allclose(f.numpy(), _f.numpy(), atol=1e-6)

    def test_log(self):
        input = np.array([[1.2, 3.3, 5.6], 
                        [9.2, 1.4, 2.3]])

        a = tinyTensor(input, dtype=dtypes.float32)
        c = a.log()

        _a = Tensor(input)
        _c = _a.log()

        assert np.allclose(c.numpy(), _c.numpy(), atol=1e-6)

    def test_mean(self):
        input = np.random.randn(5, 3)

        a = tinyTensor(input, dtype=dtypes.float32)
        c = a.mean()

        _a = Tensor(input)
        _c = _a.mean()

        assert np.allclose(c.numpy(), _c.numpy(), atol=1e-6)

    def test_exp(self):
        input = np.random.randn(5, 3)

        a = tinyTensor(input, dtype=dtypes.float32)
        c = a.exp()

        _a = Tensor(input)
        _c = _a.exp()

        assert np.allclose(c.numpy(), _c.numpy(), atol=1e-6)

    def test_add_sub_mul_div_mean_backward__normal_and_broadcast(self):
        def helper(op, input, target):
            a = tinyTensor(input, dtype=dtypes.float32, requires_grad=True)
            b = tinyTensor(target, dtype=dtypes.float32, requires_grad=True)
            if (op == 1):   c = a + b
            elif (op == 2): c = a - b
            elif (op == 3): c = a * b
            else:           c = a / b
            c.requires_grad=True
            d = c.mean(); d.requires_grad=True
            d.backward()

            _d = d.numpy(); _dg = d.grad.numpy()
            _c = c.numpy(); _cg = c.grad.numpy()
            _bg = b.grad.numpy()
            _ag = a.grad.numpy()

            a1 = Tensor(input)
            b1 = Tensor(target)
            if (op == 1):   c1 = a1 + b1
            elif (op == 2): c1 = a1 - b1
            elif (op == 3): c1 = a1 * b1
            else:           c1 = a1 / b1
            d1 = c1.mean()
            d1.backward()

            _d1 = d1.numpy(); _dg1 = d1.grad
            _c1 = c1.numpy(); _cg1 = c1.grad
            _bg1 = b1.grad
            _ag1 = a1.grad

            assert np.allclose(_d, _d1, atol=1e-6)
            assert np.allclose(_dg, _dg1, atol=1e-6)
            assert np.allclose(_c, _c1, atol=1e-6)
            assert np.allclose(_cg, _cg1, atol=1e-6)
            assert np.allclose(_bg, _bg1, atol=1e-6)
            assert np.allclose(_ag, _ag1, atol=1e-6)

        a = np.random.randn(3, 2); b = np.random.rand(3, 2)
        helper(1, a, b); helper(2, a, b); helper(3, a, b); helper(4, a, b)

        a = np.random.randn(5,3,4,1); b = np.random.rand(  3,1,1)   # broadcast
        helper(1, a, b); helper(2, a, b); helper(3, a, b); helper(4, a, b)

    def test_add_sub_mul_div_log_sum_backward__normal_and_broadcast(self):
        def helper(op, input, target):
            a = tinyTensor(input, dtype=dtypes.float32, requires_grad=True)
            b = tinyTensor(target, dtype=dtypes.float32, requires_grad=True)
            if (op == 1):   c = a + b
            elif (op == 2): c = a - b
            elif (op == 3): c = a * b
            else:           c = a / b
            c.requires_grad=True
            d = c.log(); d.requires_grad=True
            e = d.sum(); e.requires_grad=True
            e.backward()

            _e = e.numpy(); _eg = e.grad.numpy()
            _d = d.numpy(); _dg = d.grad.numpy()
            _c = c.numpy(); _cg = c.grad.numpy()
            _bg = b.grad.numpy()
            _ag = a.grad.numpy()

            a1 = Tensor(input)
            b1 = Tensor(target)
            if (op == 1):   c1 = a1 + b1
            elif (op == 2): c1 = a1 - b1
            elif (op == 3): c1 = a1 * b1
            else:           c1 = a1 / b1
            d1 = c1.log()
            e1 = d1.sum()
            e1.backward()

            _e1 = e1.numpy(); _eg1 = e1.grad
            _d1 = d1.numpy(); _dg1 = d1.grad
            _c1 = c1.numpy(); _cg1 = c1.grad
            _bg1 = b1.grad
            _ag1 = a1.grad

            assert np.allclose(_e, _e1, atol=1e-6)
            assert np.allclose(_eg, _eg1, atol=1e-6)
            assert np.allclose(_d, _d1, atol=1e-6)
            assert np.allclose(_dg, _dg1, atol=1e-6)
            assert np.allclose(_c, _c1, atol=1e-6)
            assert np.allclose(_cg, _cg1, atol=1e-6)
            assert np.allclose(_bg, _bg1, atol=1e-6)
            assert np.allclose(_ag, _ag1, atol=1e-6)
        
        a = np.random.rand(3, 2); b = a / 2
        helper(1, a, b); helper(2, a, b); helper(3, a, b); helper(4, a, b)

        # can't have negative values because of log
        a = np.random.rand(5,3,4,1) * 100; b = np.random.rand(  3,1,1) * np.min(a)  # broadcast
        helper(1, a, b); helper(2, a, b); helper(3, a, b); helper(4, a, b)
    
    def test_add_sub_mul_div_exp_sum_backward(self):
        def helper(op):
            np.random.seed(23)
            input = np.random.rand(3, 2)
            target = input / 2

            a = tinyTensor(input, dtype=dtypes.float32, requires_grad=True)
            b = tinyTensor(target, dtype=dtypes.float32, requires_grad=True)
            if (op == 1):   c = a + b
            elif (op == 2): c = a - b
            elif (op == 3): c = a * b
            else:           c = a / b
            c.requires_grad=True
            d = c.exp(); d.requires_grad=True
            e = d.sum(); e.requires_grad=True
            e.backward()

            _e = e.numpy(); _eg = e.grad.numpy()
            _d = d.numpy(); _dg = d.grad.numpy()
            _c = c.numpy(); _cg = c.grad.numpy()
            _bg = b.grad.numpy()
            _ag = a.grad.numpy()

            a1 = Tensor(input)
            b1 = Tensor(target)
            if (op == 1):   c1 = a1 + b1
            elif (op == 2): c1 = a1 - b1
            elif (op == 3): c1 = a1 * b1
            else:           c1 = a1 / b1
            d1 = c1.exp()
            e1 = d1.sum()
            e1.backward()

            _e1 = e1.numpy(); _eg1 = e1.grad
            _d1 = d1.numpy(); _dg1 = d1.grad
            _c1 = c1.numpy(); _cg1 = c1.grad
            _bg1 = b1.grad
            _ag1 = a1.grad

            assert np.allclose(_e, _e1, atol=1e-6)
            assert np.allclose(_eg, _eg1, atol=1e-6)
            assert np.allclose(_d, _d1, atol=1e-6)
            assert np.allclose(_dg, _dg1, atol=1e-6)
            assert np.allclose(_c, _c1, atol=1e-6)
            assert np.allclose(_cg, _cg1, atol=1e-6)
            assert np.allclose(_bg, _bg1, atol=1e-6)
            assert np.allclose(_ag, _ag1, atol=1e-6)
        helper(1); helper(2); helper(3); helper(4)
    
    def test_transpose(self):
        def helper(input):
            a = tinyTensor(input, dtype=dtypes.float32); ta = a.T
            _a = Tensor(input); _ta = _a.T()
            assert np.allclose(ta.numpy(), _ta.numpy(), atol=1e-6)

        a = np.random.rand(4,3); helper(a)
        a = np.random.rand(3,1); helper(a)

    def test_matmul(self):
        def helper(input1, input2):
            a1 = tinyTensor(input1, dtype=dtypes.float32)
            a2 = tinyTensor(input2, dtype=dtypes.float32)
            a3 = a1.matmul(a2)

            _a1 = Tensor(input1)
            _a2 = Tensor(input2)
            _a3 = _a1.matmul(_a2)

            assert np.allclose(a3.numpy(), _a3.numpy(), atol=1e-6)

        a = np.random.rand(4, 3); b = np.random.rand(3, 4); helper(a, b)
        a = np.random.rand(4, 3); b = np.random.rand(3, 1); helper(a, b)
        
    def test_matmul_backward(self):
        def helper(input1, input2):
            a = tinyTensor(input1, dtype=dtypes.float32, requires_grad=True)
            b = tinyTensor(input2, dtype=dtypes.float32, requires_grad=True)
            c = a.matmul(b)
            d = c.sum()
            d.backward()

            _a = Tensor(input1)
            _b = Tensor(input2)
            _c = _a.matmul(_b)
            _d = _c.sum()
            _d.backward()

            assert np.allclose(a.grad.numpy(), _a.grad, atol=1e-6)
            assert np.allclose(b.grad.numpy(), _b.grad, atol=1e-6)
            assert np.allclose(c.grad.numpy(), _c.grad, atol=1e-6)
            assert np.allclose(d.grad.numpy(), _d.grad, atol=1e-6)
            assert np.allclose(c.numpy(), _c.numpy(), atol=1e-6)
            assert np.allclose(d.numpy(), _d.numpy(), atol=1e-6)

        a = np.random.rand(3,2); b = np.random.rand(2,3); helper(a, b)
        a = np.random.rand(3,2); b = np.random.rand(2,1); helper(a, b)

    def test_add_sub_mul_div_T_mean_backward(self):
        def helper(op, input, target):
            a = tinyTensor(input, dtype=dtypes.float32, requires_grad=True)
            b = tinyTensor(target, dtype=dtypes.float32, requires_grad=True)
            if (op == 1):   c = a + b.T
            elif (op == 2): c = a - b.T
            elif (op == 3): c = a * b.T
            else:           c = a / b.T
            c.requires_grad=True
            d = c.mean(); d.requires_grad=True
            d.backward()

            _d = d.numpy(); _dg = d.grad.numpy()
            _c = c.numpy(); _cg = c.grad.numpy()
            _bg = b.grad.numpy()
            _ag = a.grad.numpy()

            a1 = Tensor(input)
            b1 = Tensor(target)
            if (op == 1):   c1 = a1 + b1.T()
            elif (op == 2): c1 = a1 - b1.T()
            elif (op == 3): c1 = a1 * b1.T()
            else:           c1 = a1 / b1.T()
            d1 = c1.mean()
            d1.backward()

            _d1 = d1.numpy(); _dg1 = d1.grad
            _c1 = c1.numpy(); _cg1 = c1.grad
            _bg1 = b1.grad
            _ag1 = a1.grad

            assert np.allclose(_d, _d1, atol=1e-6)
            assert np.allclose(_dg, _dg1, atol=1e-6)
            assert np.allclose(_c, _c1, atol=1e-6)
            assert np.allclose(_cg, _cg1, atol=1e-6)
            assert np.allclose(_bg, _bg1, atol=1e-6)
            assert np.allclose(_ag, _ag1, atol=1e-6)

        a = np.random.randn(3, 2); b = np.random.rand(2, 1)
        helper(1, a, b); helper(2, a, b); helper(3, a, b); helper(4, a, b)

        # a = np.random.randn(5,3,4,1); b = np.random.rand(  3,1,1)   # broadcast
        # helper(1, a, b); helper(2, a, b); helper(3, a, b); helper(4, a, b)
    
# a = TestTensor2D()
# a.test_add_sub_mul_div_T_mean_backward()