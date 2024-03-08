#!/usr/bin/python3

import torch
import numpy as np
from graphviz import Digraph


def trace(root):
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)
    build(root)
    return nodes, edges

def draw_dot(root):
    dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) # LR = left to right
    nodes, edges = trace(root)
    for n in nodes:
        uid = str(id(n))
        # for any value in the graph, create a rectangular ('record') node for it
        dot.node(name = uid, label = "{ %s }" % (n.label), shape='record')
        if n.op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n.op, label = n.op)
            # and connect this node to it
            dot.edge(uid + n.op, uid)
    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)
    return dot

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
    print(c)
    d = torch.tensor(   # 2x4
        [[0.0315, 0.0230, 0.0625, 0.9245],
         [0.6002, 0.0274, 0.2519, 0.3179]], requires_grad=True)
    e = c + d
    e.retain_grad()
    print(e)
    f = e.sum()
    f.retain_grad()
    print(f)

    f.backward() 

    print(f"a: {a.grad} \t\t\t {a.grad.shape}\n")
    print(f"b: {b.grad} \t\t\t {b.grad.shape}\n")
    print(f"c: {c.grad} \t\t\t {c.grad.shape}\n")
    print(f"d: {d.grad} \t\t\t {d.grad.shape}\n")
    print(f"e: {e.grad} \t\t\t {e.grad.shape}\n")
    print(f"f: {f.grad} \t\t\t {f.grad.shape}\n")

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

    # dot = draw_dot(tf)
    # dot.view()

    tf.backward() 

    print("ta: ", ta.grad)
    print("tb: ", tb.grad)
    print("tc: ", tc.grad)
    print("td: ", td.grad)
    print("te: ", te.grad)
    print("tf: ", tf.grad)

    # assert round(f.item(), 5) == round(tf.item(), 5)
    
if __name__ == '__main__':
    main()
