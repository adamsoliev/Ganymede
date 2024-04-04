# inspired by https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py

import numpy as np
from tinygrad import Tensor as __Tensor # type:ignore
from tinygrad import dtypes # type:ignore

# Today's goal
# input = np.random.randn(3, 2)
# target = np.random.rand(3, 2)
# a = Tensor(input, dtype=dtypes.float32)
# b = Tensor(target, dtype=dtypes.float32)

# c = a.sigmoid()
# loss = c.binary_crossentropy(b)
# loss.backward()

# print(loss.numpy())

class Tensor():
    def __init__(self, data):
        self.data = data

    def __repr__(self): return f"Tensor {self.data}"

    def sigmoid(self):
        return Tensor(1 / (1 + np.exp(-self.data)))
    
    def numpy(self):
        return self.data

    def binary_crossentropy(self, y):
        return Tensor(np.mean(-y.numpy() * np.log(self.data) - (1 - y.numpy()) * np.log(1 - self.data)))
    
    def log(self):
        assert np.all(self.data > 0)
        return Tensor(np.log(self.data))
    def mean(self):
        return Tensor(np.mean(self.data))
    def __neg__(self): 
        return Tensor(-self.data)
    def __add__(self, x):
        return Tensor(self.data + x.numpy())
    def __sub__(self, x):
        return self + -x
    def __mul__(self, x):
        return Tensor(self.data * x.numpy())


def test_bce():
    input = np.random.randn(3, 2)
    target = np.random.rand(3, 2)

    a = __Tensor(input, dtype=dtypes.float32)
    b = __Tensor(target, dtype=dtypes.float32)
    c = a.sigmoid()
    loss = c.binary_crossentropy(b)

    _a = Tensor(input)
    _b = Tensor(target)
    _c = _a.sigmoid()
    _loss = _c.binary_crossentropy(_b)

    assert np.allclose(c.numpy(), _c.numpy(), atol=1e-6)
    assert np.allclose(loss.numpy(), _loss.numpy(), atol=1e-6)

def test_add_sub_mul():
    input = np.random.randn(3, 2)
    target = np.random.rand(3, 2)

    a = __Tensor(input, dtype=dtypes.float32)
    b = __Tensor(target, dtype=dtypes.float32)
    c = a + b
    d = a - b
    e = a * b

    _a = Tensor(input)
    _b = Tensor(target)
    _c = _a + _b
    _d = _a - _b
    _e = _a * _b

    assert np.allclose(c.numpy(), _c.numpy(), atol=1e-6)
    assert np.allclose(d.numpy(), _d.numpy(), atol=1e-6)
    assert np.allclose(e.numpy(), _e.numpy(), atol=1e-6)

def test_log():
    input = np.array([[1.2, 3.3, 5.6], 
                      [9.2, 1.4, 2.3]])

    a = __Tensor(input, dtype=dtypes.float32)
    c = a.log()

    _a = Tensor(input)
    _c = _a.log()

    assert np.allclose(c.numpy(), _c.numpy(), atol=1e-6)

def test_mean():
    input = np.random.randn(5, 3)

    a = __Tensor(input, dtype=dtypes.float32)
    c = a.mean()

    _a = Tensor(input)
    _c = _a.mean()

    assert np.allclose(c.numpy(), _c.numpy(), atol=1e-6)


def main():
    np.random.seed(23)
    test_bce()
    test_add_sub_mul()
    test_log()


if __name__ == "__main__":
    main()