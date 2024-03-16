#!/usr/bin/python3

import torch
from torch import Tensor
import numpy as np
from torch import nn
from matplotlib import pyplot as plt    # type: ignore
from sklearn import datasets            # type: ignore

class NN(nn.Module):
    def __init__(self, input_size: int, H1: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, H1)
        self.linear2 = nn.Linear(H1, output_size)
    
    def forward(self, x: 'Tensor') -> 'Tensor':
        x = torch.sigmoid(self.linear(x))
        x = torch.sigmoid(self.linear2(x))
        return x
    
def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor) -> None:
    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()
    y_prob = model.forward(X_to_pred_on).squeeze()
    y_pred = torch.round(y_prob)  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

def main() -> None:
    n_pts = 1000
    X, y = datasets.make_circles(n_samples=n_pts, random_state=42, noise=0.04)
    x_data = torch.FloatTensor(X)
    y_data = torch.FloatTensor(y.reshape(1000, 1))

    torch.manual_seed(2)
    model = NN(2, 10, 1)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 1000
    losses = []
    for epoch in range(epochs):
        y_pred = model.forward(x_data).squeeze()
        loss = loss_fn(y_pred, y_data.squeeze())
        print(f"epoch: {epoch}, loss: {loss.item()}")
        
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Plot decision boundaries for training and test sets
    plt.figure(figsize=(6, 6))
    plt.title("Train")
    plot_decision_boundary(model, x_data, y_data)
    plt.show()

if __name__ == '__main__':
    main()
