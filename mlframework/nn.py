import random
import numpy as np
from tensor import Tensor

class Module:
    def zero_grad(self) -> None:
        for p in self.parameters():
            p.grad = np.array(0.0)
    
    def parameters(self) -> list['Tensor']:
        return []

# class Neuron(Module):
#     def __init__(self, nin: int, nonlin: bool=True) -> None:
#         self.w = [Tensor(random.uniform(-1,1)) for _ in range(nin)]
#         self.b = Tensor(0.0)
#         self.nonlin = nonlin
    
#     def __call__(self, x: 'Tensor') -> 'Tensor':
#         act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
#         return act.relu() if self.nonlin else act
    
#     def parameters(self) -> list['Tensor']:
#         return self.w + [self.b]
    
#     def __repr__(self) -> str:
#         return f"{'Relu' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

# class Layer(Module):
#     def __init__(self, nin: int, nout: int, **kwargs: bool) -> None:
#         self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]
    
#     def __call__(self, x: 'Tensor') -> list['Tensor']:
#         return [n(x) for n in self.neurons]
    
#     def parameters(self) -> list['Tensor']:
#         return [p for n in self.neurons for p in n.parameters()]
    
#     def __repr__(self) -> str:
#         return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"
    

# class MLP(Module):
#     def __init__(self, nin: int, nouts: list[int]) -> None:
#         sz = [nin] + nouts
#         self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]
    
#     def __call__(self, x: 'Tensor') -> 'Tensor':
#         for layer in self.layers:
#             x = layer(x)
#         return x
    
#     def parameters(self) -> list['Tensor']:
#         return [p for layer in self.layers for p in layer.parameters()]

#     def __repr__(self) -> str:
#         return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
