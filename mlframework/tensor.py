# inspired by https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py

import numpy as np
from utils import gen_label, prod
from enum import Enum, auto

class BinaryOps(Enum): ADD = auto(); SUB = auto(); MUL = auto(); DIV = auto()
class UnaryOps(Enum): NEG = auto(); LOG = auto(); EXP = auto() 
class ReduceOps(Enum): MEAN = auto(); SUM = auto()

class Tensor():
    def __init__(self, data, children=set(), op=""):
        assert isinstance(data, (np.ndarray, int, float))
        if isinstance(data, (int, float)): 
            data = np.array([float(data)], dtype=np.float32)

        data.astype(np.float32)
        self.data = data
        self.prev = set(children)
        self._backward = lambda: None
        self.label = gen_label()
        self.op = op
        self.grad = 0
        self.shape = data.shape

    def __repr__(self): return f"Tensor {self.data}"

    def __format__(self, format_spec):
        assert isinstance(self.data, np.ndarray)
        return format(self.data.item(), format_spec)  
    
    def numpy(self): return self.data

    # *** functional ops ***
    def binary_crossentropy(self, y):
        return (-y*self.log() - (1-y)*(1-self).log()).mean()

    # *** reduce ops ***
    def mean(self): return e([self], ReduceOps.MEAN)
    def sum(self): return e([self], ReduceOps.SUM)
    
    # *** mlops (unary) ***
    def log(self): return e([self], UnaryOps.LOG)
    def exp(self): return e([self], UnaryOps.EXP)
    # https://stackoverflow.com/questions/40726490/overflow-error-in-pythons-numpy-exp-function
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

    def T(self):
        result = Tensor(np.transpose(self.data), {self, }, "T")
        def _backward():
            self.grad += np.transpose(result.grad)
        result._backward = _backward
        return result

    def matmul(self, x):
        result = Tensor(np.matmul(self.numpy(), x.numpy()), {self, x}, "MATMUL") 
        def _backward():
            self.grad += result.grad @ x.numpy().T
            x.grad += self.data.T @ result.grad 
        result._backward = _backward
        return result
    
    def squeeze(self):
        return self.reshape(tuple(dim for dim in self.shape if dim != 1))
    
    def reshape(self, shape):
        result = Tensor(self.numpy().reshape(shape), {self, }, "RESHAPE")
        def _backward():
            self.grad += result.grad
        result._backward = _backward
        return result
    
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

        self.grad = np.ones_like(self.data, dtype=np.float32)
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
                def grad_helper(result, node): # broadcasting
                    grad = np.copy(result.grad)
                    ndims_added = grad.ndim - node.numpy().ndim
                    for _ in range(ndims_added):
                        grad = grad.sum(axis=0)
                    for i, dim in enumerate(node.numpy().shape):
                        if dim == 1:
                            grad = grad.sum(axis=i, keepdims=True)
                    return grad; 
                left.grad += grad_helper(result, left)
                right.grad += grad_helper(result, right)
            result._backward = _backward
            return result
        elif (op == BinaryOps.SUB): 
            result = Tensor(left.numpy() - right.numpy(), {left, right}, op.name)
            def _backward():
                def grad_helper(result, node): # broadcasting
                    grad = np.copy(result.grad)
                    ndims_added = grad.ndim - node.numpy().ndim
                    for _ in range(ndims_added):
                        grad = grad.sum(axis=0)
                    for i, dim in enumerate(node.numpy().shape):
                        if dim == 1:
                            grad = grad.sum(axis=i, keepdims=True)
                    return grad; 
                left.grad += grad_helper(result, left)
                right.grad += -grad_helper(result, right)
            result._backward = _backward
            return result
        elif (op == BinaryOps.MUL): 
            result = Tensor(left.numpy() * right.numpy(), {left, right}, op.name)
            def _backward():
                def grad_helper(result, node, other): # broadcasting
                    grad = np.copy(result.grad)
                    grad = grad * other.numpy()
                    ndims_added = grad.ndim - node.numpy().ndim
                    for _ in range(ndims_added):
                        grad = grad.sum(axis=0)
                    for i, dim in enumerate(node.numpy().shape):
                        if dim == 1:
                            grad = grad.sum(axis=i, keepdims=True)
                    return grad; 
                left.grad += grad_helper(result, left, right)
                right.grad += grad_helper(result, right, left)
            result._backward = _backward
            return result
        elif (op == BinaryOps.DIV): 
            result = Tensor(left.numpy() / right.numpy(), {left, right}, op.name)
            def _backward():
                def grad_helper(result, node, other, isNumerator): # broadcasting
                    grad = np.copy(result.grad)
                    if isNumerator:
                        grad = grad / other.numpy()
                    else:
                        grad = grad * (-other.numpy() / (node.numpy() * node.numpy()))
                    ndims_added = grad.ndim - node.numpy().ndim
                    for _ in range(ndims_added):
                        grad = grad.sum(axis=0)
                    for i, dim in enumerate(node.numpy().shape):
                        if dim == 1:
                            grad = grad.sum(axis=i, keepdims=True)
                    return grad; 
                left.grad += grad_helper(result, left, right, True)
                right.grad += grad_helper(result, right, left, False)
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
                left.grad += np.full((left.shape), result.grad) / prod(left.shape)
            result._backward = _backward
            return result
        elif (op == ReduceOps.SUM): 
            result = Tensor(np.sum(left.numpy()), {left, }, op.name)
            def _backward():
                left.grad += np.full((left.shape), result.grad)
            result._backward = _backward
            return result
        else: assert 0
    else:
        assert 0
