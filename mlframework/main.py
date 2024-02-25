"""
first implement deep neural networks with numpy, no framework at all
if interested in machine learning, the scikit-learn docs actually make for great notes
"""

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import math
import numpy as np
import matplotlib.pyplot as plt
import torch
from graphviz import Digraph

def trace(root):
    # builds a set of all nodes and edges in a graph
    nodes, edges = set(), set()
    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
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
        dot.node(name = uid, label = "{ %s | data %.4f | grad %.4f }" % (n.label, n.data, n.grad), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out
    
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    
    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other
    
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')
        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)
        out = Value(t, (self, ), 'tanh')
        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out
    
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

import random

class Neuron:
    def __init__(self, nin):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]
        self.b = Value(random.uniform(-1, 1))
    
    def __call__(self, x):
        act = sum(wi * xi for wi, xi in zip(self.w, x)) + self.b
        out = act.tanh()
        return out

def main():
    x = [2.0, 3,0, 9.0]
    n = Neuron(3)
    print(n(x))
    # inputs
    # x1 = Value(2.0, label='x1')
    # x2 = Value(0.0, label='x2')
    # # weights
    # w1 = Value(-3.0, label='w1')
    # w2 = Value(1.0, label='w2')
    # # bias
    # b = Value(6.881373587, label='b')
    # # x1 * w1 + x2 * w2 + b
    # x1w1 = x1 * w1; x1w1.label = 'x1w1'
    # x2w2 = x2 * w2; x2w2.label = 'x2w2'
    # x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1w1x2w2'
    # n = x1w1x2w2 + b; n.label = 'n'
    # o = n.tanh(); o.label = 'o'
    # o.backward()

    # dot = draw_dot(o)
    # dot.view()

    # a = Value(-2.0, label='a')
    # b = Value(3.0, label='b')
    # d = a * b ; d.label = 'd'
    # e = a + b ; e.label = 'e'
    # f = d * e ; f.label = 'f'
    # f.backward()
    # dot1 = draw_dot(f)
    # dot1.view()

    """ ABOVE IMPLEMENTED IN PYTORCH """
    # x1 = torch.Tensor([2.0]).double()         ; x1.requires_grad = True 
    # x2 = torch.Tensor([0.0]).double()         ; x2.requires_grad = True     
    # w1 = torch.Tensor([-3.0]).double()        ; w1.requires_grad = True         
    # w2 = torch.Tensor([1.0]).double()         ; w2.requires_grad = True 
    # b = torch.Tensor([6.881373587]).double()  ; b.requires_grad = True     
    # n = x1 * w1 + x2 * w2 + b
    # o = torch.tanh(n)

    # print(o.data.item())
    # o.backward()
    # print("------------")
    # print("x1 ", x1.grad.item())
    # print("x2 ", x2.grad.item())
    # print("w1 ", w1.grad.item())
    # print("w2 ", w2.grad.item())

if __name__ == '__main__':
    main()

"""
    into neuron:
        (input * weight)s
    in neuron
        sum of (input * weight)s + bias
    out of neuron
        activation function applied to (sum of (input * weight)s + bias)
"""