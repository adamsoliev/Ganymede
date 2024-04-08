#!/usr/bin/python3

import sys
sys.path.append('../')

import torch
from tensor import Tensor
import numpy as np
from matplotlib import pyplot as plt    # type: ignore
from sklearn import datasets            # type: ignore
from sklearn.model_selection import train_test_split
from utils import prod

# Hyperparameters
HL = 20
EPOCHS = 2000
LR = 0.001

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")
    X, y = X.to("cpu"), y.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_prob_distr = model(X_to_pred_on)

    y_pred = torch.round(y_prob_distr)  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

import math

class Linear():
    def __init__(self, nin: int, nout: int) -> None:
        # kaiming_uniform
        fan  = nin
        gain = math.sqrt(2.0 / (1 + 2.23606797749**2)) # constant corresponds to leaky_relu
        std = gain / math.sqrt(fan)
        bound = math.sqrt(3.0) * std
        initw = np.random.uniform(-bound, bound, nout * nin).reshape(nout, nin)
        bound = 1 / math.sqrt(fan)
        initb = np.random.uniform(-bound, bound, nout)

        self.w = Tensor(initw)
        self.b = Tensor(initb)
    
    def __call__(self, x: 'Tensor') -> 'Tensor':
        return x.matmul(self.w.T()) + self.b

    def parameters(self) -> list['Tensor']:
        return [self.w, self.b]

# class NN(nn.Module):
#     def __init__(self, input_size: int, H1: int, output_size: int):
#         super().__init__()
#         self.linear = nn.Linear(input_size, H1)
#         self.sigmoid1 = nn.Sigmoid()
#         self.linear2 = nn.Linear(H1, output_size)
#         self.sigmoid2 = nn.Sigmoid()
    
#     def forward(self, x: 'Tensor') -> 'Tensor':
#         x = self.linear(x)
#         x = self.sigmoid1(x)
#         x = self.linear2(x)
#         x = self.sigmoid2(x)
#         return x

in1 = np.random.rand(4,5)

pylinear = torch.nn.Linear(5, 2)
pyt = torch.Tensor(in1)
b = pylinear(pyt)

t = Tensor(in1)
linear = Linear(5, 2)
a = linear(t)

print(f"output shapes: {a.shape} vs {b.shape}")
print("Outputs: ")
print(b)
print(a)
# print(linear.parameters())
# print("=============")
# print(b)
# for _, __ in pylinear.named_parameters():
#     print(_, __)


