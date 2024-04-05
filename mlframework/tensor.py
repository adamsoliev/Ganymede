# inspired by https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py

import numpy as np
from tinygrad import Tensor as __Tensor # type:ignore
from tinygrad import dtypes # type:ignore
from utils import draw_dot, gen_label

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
    def __init__(self, data, children=set(), op=""):
        assert isinstance(data, (np.ndarray, int, float))
        if isinstance(data, int): data = float(data)

        self.data = data
        self.prev = set(children)
        self._backward = lambda: None
        self.label = gen_label()
        self.op = op

    def __repr__(self): return f"Tensor {self.data}"
    
    def numpy(self):
        return self.data

    # *** functional ops ***
    def binary_crossentropy(self, y):
        return (-y*self.log() - (1-y)*(1-self).log()).mean()

    # *** reduce ops ***
    def mean(self):
        return Tensor(np.mean(self.data), {self, }, "mean")
    
    # *** mlops (unary) ***
    def log(self):
        assert np.all(self.data > 0)
        return Tensor(np.log(self.data), {self, }, "log")
    def exp(self):
        return Tensor(np.exp(self.data), {self, }, "exp")
    def sigmoid(self):
        return Tensor((1 / (1 + (-self).exp())).numpy(), {self, }, "sigmoid")

    # *** op wrappers ***
    def __neg__(self): 
        return Tensor(-self.data, {self, }, "neg")
    def __add__(self, x):
        if isinstance(x, int): x = Tensor(x)
        return Tensor(self.data + x.numpy(), {self, x}, "add")
    def __sub__(self, x):
        if isinstance(x, int): x = Tensor(x)
        return self + -x
    def __mul__(self, x):
        if isinstance(x, int): x = Tensor(x)
        return Tensor(self.data * x.numpy(), {self, x}, "mul")
    def __truediv__(self, x):
        if isinstance(x, int): x = Tensor(x)
        return Tensor(self.numpy() / x.numpy(), {self, x}, "div")

    def __radd__(self, x):
        if isinstance(x, int): x = Tensor(x)
        return x + self
    def __rsub__(self, x):
        if isinstance(x, int): x = Tensor(x)
        return x - self
    def __rmul__(self, x):
        if isinstance(x, int): x = Tensor(x)
        return x * self
    def __rtruediv__(self, x):
        if isinstance(x, int): x = Tensor(x)
        return x / self
    
    def backward(self) -> None:
        topsorted = []
        visited = set()
        def helper_topsort(v: 'Tensor') -> None:
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    helper_topsort(child)
                topsorted.append(v)
        helper_topsort(self)

        # self.grad = np.ones_like(self.data)
        for node in reversed(topsorted):
            node._backward()

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

    dot = draw_dot(_loss)
    dot.view()

    assert np.allclose(c.numpy(), _c.numpy(), atol=1e-6)
    assert np.allclose(loss.numpy(), _loss.numpy(), atol=1e-6)

def test_add_sub_mul_div():
    input = np.random.randn(3, 2)
    target = np.random.rand(3, 2)

    a = __Tensor(input, dtype=dtypes.float32)
    b = __Tensor(target, dtype=dtypes.float32)
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
    test_add_sub_mul_div()
    test_log()
    test_mean()

if __name__ == "__main__":
    main()