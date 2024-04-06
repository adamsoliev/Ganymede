from graphviz import Digraph # type:ignore

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
        # dot.node(name = uid, label = "{ %s }" % (n.label), shape='record')
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

def gen_label(name="T"):
    gen_label.cnt = getattr(gen_label, 'cnt', 0) + 1
    return f"{name}{gen_label.cnt}"

import functools, operator
def prod(x):
    return functools.reduce(operator.mul, x, 1)