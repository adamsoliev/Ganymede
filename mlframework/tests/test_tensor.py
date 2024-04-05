import unittest

import numpy as np
from tinygrad import Tensor as tinyTensor # type:ignore
from tinygrad import dtypes # type:ignore
from utils import draw_dot, gen_label
from tensor import Tensor

class TestTensor(unittest.TestCase):
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

