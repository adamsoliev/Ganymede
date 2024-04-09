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
import math

# Hyperparameters
HL = 20
EPOCHS = 1000
LR = 0.01

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
        y_prob_distr = model.forward(X_to_pred_on)

    y_pred = torch.round(y_prob_distr)  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


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

class SGD:
    def __init__(self, params: list['Tensor'], lr: float = 0.01) -> None:
        self.learning_rate = lr
        self.params = params
    
    def step(self) -> None:
        for param in self.params:
            param.data -= param.grad * self.learning_rate
    
    def zero_grad(self) -> None:
        for param in self.params:
            param.grad = 0

class NN():
    def __init__(self, input_size: int, H1: int, output_size: int):
        self.linear = Linear(input_size, H1)
        self.linear2 = Linear(H1, output_size)
    
    def forward(self, x: 'Tensor') -> 'Tensor':
        x = self.linear(x)
        x = x.sigmoid()
        x = self.linear2(x)
        x = x.sigmoid()
        return x
    
    def parameters(self):
        params = []
        for _ in self.linear.parameters():
            params.append(_)
        for _ in self.linear2.parameters():
            params.append(_)
        return params

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def main() -> None:
    n_pts = 1000
    X, y = datasets.make_circles(n_samples=n_pts, random_state=42, noise=0.04)

    # plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    # plt.show()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42) # make the random split reproducible
    X_train = Tensor(X_train)
    X_test = Tensor(X_test)
    y_train = Tensor(y_train)
    y_test = Tensor(y_test)

    # usual stuff
    model = NN(2, HL, 1)
    # loss_fn = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=LR)

    epochs = EPOCHS

    for epoch in range(epochs):
        y_prob_distr = model.forward(X_train).squeeze()
        loss = y_prob_distr.binary_crossentropy(y_train)
        # acc = accuracy_fn(y_true=y_train, y_pred=torch.round(y_prob_distr))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Testing
        test_prob_distr = model.forward(X_test).squeeze()
        test_loss = test_prob_distr.binary_crossentropy(y_test)
        # test_acc = accuracy_fn(y_true=y_test, y_pred=torch.round(test_prob_distr))

        if epoch % 10 == 0:
            # print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Test loss: {test_loss}")
    
    # assert test_acc > 90.0

    # visualize 
    # plt.figure(figsize=(12, 6))
    # plt.subplot(1, 2, 1)
    # plt.title("Train")
    # plot_decision_boundary(model, X_train, y_train)
    # plt.subplot(1, 2, 2)
    # plt.title("Test")
    # plot_decision_boundary(model, X_test, y_test)
    # plt.show()

if __name__ == '__main__':
    main()
