# inspired by https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py

import numpy as np
from utils import gen_label
from enum import Enum, auto

class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto()
class UnaryOps(Enum): NEG = auto(); LOG = auto(); EXP = auto() 
class ReduceOps(Enum): MEAN = auto()

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
        self.grad = 0

    def __repr__(self): return f"Tensor {self.data}"
    
    def numpy(self): return self.data

    # *** functional ops ***
    def binary_crossentropy(self, y):
        return (-y*self.log() - (1-y)*(1-self).log()).mean()

    # *** reduce ops ***
    def mean(self): return e([self], ReduceOps.MEAN)
    
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

        self.grad = np.ones_like(self.data)
        for node in reversed(topsorted):
            node._backward()


def e(srcs, op):
    if isinstance(op, UnaryOps):
        assert isinstance(srcs, list) and len(srcs) == 1
        left = srcs[0]
        if isinstance(left, (float, int)): left = Tensor(np.array([left]))
        if (op == UnaryOps.NEG): 
            result = Tensor(-left.numpy(), {left, }, op.name)
            def _backward():
                left.grad += -result.grad
            result._backward = _backward
            return result
        elif (op == UnaryOps.LOG): 
            assert np.all(left.numpy() > 0)
            result = Tensor(np.log(left.numpy()), {left, }, op.name)
            def _backward():
                left.grad += result.grad * (1 / left.numpy())
            result._backward = _backward
            return result
        elif (op == UnaryOps.EXP): 
            result = Tensor(np.exp(left.numpy()), {left, }, op.name)
            def _backward():
                left.grad += result.grad * result.numpy()
            result._backward = _backward
            return result
        else: assert 0

    elif isinstance(op, BinaryOps):
        assert isinstance(srcs, list) and len(srcs) == 2
        left, right = srcs[0], srcs[1]
        if isinstance(left, (float, int)): left = Tensor(np.array([left]))
        if isinstance(right, (float, int)): right = Tensor(np.array([right]))
        if (op == BinaryOps.ADD): 
            result = Tensor(left.numpy() + right.numpy(), {left, right}, op.name)
            def _backward():
                left.grad += result.grad; right.grad += result.grad 
            result._backward = _backward
            return result
        elif (op == BinaryOps.SUB): 
            result = Tensor(left.numpy() - right.numpy(), {left, right}, op.name)
            def _backward():
                left.grad += result.grad; right.grad += -result.grad
            result._backward = _backward
            return result
        elif (op == BinaryOps.MUL): 
            result = Tensor(left.numpy() * right.numpy(), {left, right}, op.name)
            def _backward():
                left.grad += result.grad * right.numpy(); right.grad += result.grad * left.numpy() 
            result._backward = _backward
            return result
        elif (op == BinaryOps.DIV): 
            result = Tensor(left.numpy() / right.numpy(), {left, right}, op.name)
            def _backward():
                left.grad += result.grad / right.numpy(); right.grad += result.grad * (-left.numpy() / (right.numpy() * right.numpy()))
            result._backward = _backward
            return result
        else: assert 0

    elif isinstance(op, ReduceOps):
        assert isinstance(srcs, list) and len(srcs) == 1
        left = srcs[0]
        if isinstance(left, (float, int)): left = Tensor(np.array([left]))
        if (op == ReduceOps.MEAN): 
            result = Tensor(np.mean(left.numpy()), {left, }, op.name)
            def _backward():
                left.grad += np.full((left.numpy().shape[0], left.numpy().shape[1]), result.grad) / (left.numpy().shape[0] * left.numpy().shape[1])
            result._backward = _backward
            return result
        else: assert 0
    else:
        assert 0
