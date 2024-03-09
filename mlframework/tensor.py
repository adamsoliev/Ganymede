#!/usr/bin/python3

import numpy as np
from numpy.typing import NDArray
from typing import Callable

class Tensor:
    labelnum = 1

    def __init__(self, 
                 data: NDArray[np.float64], 
                 children: set['Tensor'] = set(), 
                 op: str = "") -> None:
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, (int, float)):
            self.data = np.array([data], dtype=np.float64)
        else:
            assert isinstance(data, list)
            self.data = np.array(data, dtype=np.float64)
        self.prev = set(children)
        self.op = op

        def genlabel() -> str:
            name = f"T{Tensor.labelnum}"
            Tensor.labelnum += 1
            return name
        self.label = genlabel()
        self._backward: Callable[[], None] = lambda: None
        self.grad = np.zeros_like(self.data)
    
    def item(self) -> np.float64:
        assert len(self.data.shape) == 1    
        assert self.data.shape[0] == 1 
        return self.data.flat[0]
    
    def sum(self) -> 'Tensor':
        result = Tensor(self.data.sum(), {self, }, "sum")
        def _backward() -> None:
            self.grad += np.ones_like(self.data) * result.grad
        result._backward = _backward
        return result
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        result = Tensor(np.matmul(self.data, other.data), {self, other}, "matmul")
        def _backward() -> None:
            self.grad += result.grad @ other.data.T
            other.grad += self.data.T @ result.grad
        result._backward = _backward
        return result

    def __add__(self, other: 'Tensor') -> 'Tensor':
        result = Tensor(self.data + other.data, {self, other}, "+")
        def _backward() -> None:
            # self
            grad = np.copy(result.grad)
            ndims_added = grad.ndim - self.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            
            for i, dim in enumerate(self.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            self.grad += grad

            # other
            grad = np.copy(result.grad)
            ndims_added = grad.ndim - other.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            
            for i, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            other.grad += grad
            
        result._backward = _backward
        return result
    
    def __mul__(self, other: 'Tensor') -> 'Tensor':
        result = Tensor(self.data * other.data, {self, other}, "*")
        def _backward() -> None:
            # self
            grad = np.copy(result.grad)

            grad = grad * other.data
            ndims_added = grad.ndim - self.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            
            for i, dim in enumerate(self.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            self.grad += grad

            # other
            grad = np.copy(result.grad)

            grad = grad * self.data
            ndims_added = grad.ndim - other.data.ndim
            for _ in range(ndims_added):
                grad = grad.sum(axis=0)
            
            for i, dim in enumerate(other.data.shape):
                if dim == 1:
                    grad = grad.sum(axis=i, keepdims=True)
            other.grad += grad

        result._backward = _backward
        return result

    def __repr__(self) -> str:
        return f"Tensor: \n{self.data}"
    
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