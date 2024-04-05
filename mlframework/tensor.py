# inspired by https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py

import numpy as np
from tinygrad import Tensor as __Tensor # type:ignore
from tinygrad import dtypes # type:ignore
from utils import draw_dot, gen_label

from enum import Enum, auto
class BinaryOps(Enum):
    ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto()
class UnaryOps(Enum):
    NEG = auto(); LOG = auto(); EXP = auto() 
class ReduceOps(Enum):
    MEAN = auto()

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
        if isinstance(data, (int, float)): 
            data = np.array([float(data)])

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
    def mean(self): return e([self], ReduceOps.MEAN)
        # return Tensor(np.mean(self.data), {self, }, "mean")
    
    # *** mlops (unary) ***
    def log(self): return e([self], UnaryOps.LOG)
    def exp(self): return e([self], UnaryOps.EXP)
    def sigmoid(self): return 1 / (1 + (-self).exp())

    # *** op wrappers ***
    def __neg__(self): return e([self], UnaryOps.NEG)

    def __add__(self, x): return e([self, x], BinaryOps.ADD)
    def __sub__(self, x): return e([self, x], BinaryOps.SUB)
    def __mul__(self, x): return e([self, x], BinaryOps.MUL)
    def __truediv__(self, x): return e([self, x], BinaryOps.DIV)

    def __radd__(self, x): return e([x, self], BinaryOps.ADD)
    def __rsub__(self, x): return e([x, self], BinaryOps.SUB)
    def __rmul__(self, x): return e([x, self], BinaryOps.MUL)
    def __rtruediv__(self, x): return e([x, self], BinaryOps.DIV)
    
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


def e(srcs, op):
    if isinstance(op, UnaryOps):
        assert isinstance(srcs, list) and len(srcs) == 1
        left = srcs[0]
        if isinstance(left, (float, int)): left = Tensor(np.array([left]))
        if (op == UnaryOps.NEG): return Tensor(-left.numpy(), {left, }, op.name)
        elif (op == UnaryOps.LOG): 
            assert np.all(left.numpy() > 0)
            return Tensor(np.log(left.numpy()), {left, }, op.name)
        elif (op == UnaryOps.EXP): return Tensor(np.exp(left.numpy()), {left, }, op.name)
        else: assert 0

    elif isinstance(op, BinaryOps):
        assert isinstance(srcs, list) and len(srcs) == 2
        left, right = srcs[0], srcs[1]
        if isinstance(left, (float, int)): left = Tensor(np.array([left]))
        if isinstance(right, (float, int)): right = Tensor(np.array([right]))
        if (op == BinaryOps.ADD): return Tensor(left.numpy() + right.numpy(), {left, right}, op.name)
        elif (op == BinaryOps.SUB): return Tensor(left.numpy() - right.numpy(), {left, right}, op.name)
        elif (op == BinaryOps.MUL): return Tensor(left.numpy() * right.numpy(), {left, right}, op.name)
        elif (op == BinaryOps.DIV): return Tensor(left.numpy() / right.numpy(), {left, right}, op.name)
        else: assert 0

    elif isinstance(op, ReduceOps):
        assert isinstance(srcs, list) and len(srcs) == 1
        left = srcs[0]
        if isinstance(left, (float, int)): left = Tensor(np.array([left]))
        if (op == ReduceOps.MEAN): return Tensor(np.mean(left.numpy()), {left, }, op.name)
        else: assert 0
    else:
        assert 0

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

    # dot = draw_dot(_loss)
    # dot.view()

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

def test_radd_rsub_rmul_rdiv():
    input = np.random.rand(3, 2)

    b = __Tensor(input, dtype=dtypes.float32)
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
    test_radd_rsub_rmul_rdiv()
    test_log()
    test_mean()

if __name__ == "__main__":
    main()