"""
first implement deep neural networks with numpy, no framework at all
if interested in machine learning, the scikit-learn docs actually make for great notes
"""

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import math
import numpy as np
import matplotlib.pyplot as plt

from graphviz import Digraph


class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self._prev = set(_children)
        self._op = _op
        self.label = label
    def __repr__(self):
        return f"Value(data={self.data})"
    def __add__(self, other):
        return Value(self.data + other.data, (self, other), '+')
    def __mul__(self, other):
        if other.__class__ == float:
             self.data = self.data * other
             return self
        return Value(self.data * other.data, (self, other), '*')
    

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
        dot.node(name = uid, label = "{ %s | data %.4f }" % (n.label, n.data), shape='record')
        if n._op:
            # if this value is a result of some operation, create an op node for it
            dot.node(name = uid + n._op, label = n._op)
            # and connect this node to it
            dot.edge(uid + n._op, uid)

    for n1, n2 in edges:
        # connect n1 to the op node of n2
        dot.edge(str(id(n1)), str(id(n2)) + n2._op)

    return dot

def main():
    a = Value(2.0, label='a')
    b = Value(4.0, label='b')
    c = a + b; c.label = 'c'
    d = Value(-2.9, label='d')
    e = c * 2.0 + d; e.label = 'e'
    dot = draw_dot(e)
    dot.view()
    # print(e._prev)
    # print(e._op)

if __name__ == '__main__':
    main()