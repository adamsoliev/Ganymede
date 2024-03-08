#!/usr/bin/python3

import torch
import numpy as np

class Tensor:
    labelnum = 1

    def __init__(self, data, children=(), op=""):
        if type(data) == np.ndarray:
            self.data = data
        elif isinstance(data, (int, float)):
            self.data = np.array([data])
        else:
            assert type(data) == list
            self.data = np.array(data)
        self.prev = set(children)
        self.op = op

        def genlabel():
            name = f"T{Tensor.labelnum}"
            Tensor.labelnum += 1
            return name
        self.label = genlabel()
        self._backward = lambda: None
        self.grad = np.zeros_like(self.data)
    
    def item(self):
        assert len(self.data.shape) == 1    
        assert self.data.shape[0] == 1 
        return self.data[0]
    
    def sum(self):
        result = Tensor(self.data.sum(), (self, ), "sum")
        def _backward():
            self.grad += np.ones_like(self.data) * result.grad
        result._backward = _backward
        return result
    
    def matmul(self, other):
        assert type(other) == Tensor
        result = Tensor(np.matmul(self.data, other.data), (self, other), "matmul")
        def _backward():
            self.grad += result.grad @ other.data.T
            other.grad += self.data.T @ result.grad
        result._backward = _backward
        return result

    def __add__(self, other):
        assert type(other) == Tensor
        result = Tensor(self.data + other.data, (self, other), "+")
        def _backward():
            self.grad += np.ones_like(self.data) * result.grad
            other.grad += np.ones_like(self.data) * result.grad
        result._backward = _backward
        return result

    def __repr__(self):
        return f"Tensor: \n{self.data}"
    
    def backward(self):
        topsorted = []
        visited = set()
        def helper_topsort(v):
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    helper_topsort(child)
                topsorted.append(v)
        helper_topsort(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topsorted):
            node._backward()
    

def main():
    # ------------------ PYTORCH ------------------
    torch.set_printoptions(precision=9)

    a = torch.tensor([[0.2606, 0.0398, 0.2312], [0.4034, 0.8265, 0.7248]], requires_grad=True)
    b = torch.tensor(   # 3x4
        [[0.2026, 0.4692, 0.6961, 0.0221],
         [0.7270, 0.7451, 0.8819, 0.2733],
         [0.8547, 0.2478, 0.0153, 0.8785]], requires_grad=True)
    c = a.matmul(b)
    c.retain_grad()
    d = torch.tensor(   # 2x4
        [[0.0315, 0.0230, 0.0625, 0.9245],
         [0.6002, 0.0274, 0.2519, 0.3179]], requires_grad=True)
    e = c + d
    e.retain_grad()
    f = e.sum()
    f.retain_grad()

    f.backward() 

    print(f"a: {a.grad} \t\t\t {a.grad.shape}")
    print(f"b: {b.grad} \t\t\t {b.grad.shape}")
    print(f"c: {c.grad} \t\t\t {c.grad.shape}")
    print(f"d: {d.grad} \t\t\t {d.grad.shape}")
    print(f"e: {e.grad} \t\t\t {e.grad.shape}")
    print(f"f: {f.grad} \t\t\t {f.grad.shape}")

    # ------------------ OURTORCH ------------------
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

    print(f"ta: {ta.grad} \t\t\t {ta.grad.shape}")
    print(f"tb: {tb.grad} \t\t\t {tb.grad.shape}")
    print(f"tc: {tc.grad} \t\t\t {tc.grad.shape}")
    print(f"td: {td.grad} \t\t\t {td.grad.shape}")
    print(f"te: {te.grad} \t\t\t {te.grad.shape}")
    print(f"tf: {tf.grad} \t\t\t {tf.grad.shape}")

    assert round(f.item(), 5) == round(tf.item(), 5)
    
if __name__ == '__main__':
    main()
