#!/usr/bin/python3

import torch
import numpy as np
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

class NN(nn.Module):
    def __init__(self, input_size, H1, output_size):
        super().__init__()
        self.linear = nn.Linear(input_size, H1)
        self.linear2 = nn.Linear(H1, output_size)
    
    def forward(self, x):
        x = torch.sigmoid(self.linear(x))
        x = torch.sigmoid(self.linear2(x))
        return x
    
    def predict(self, x):
        pred = self.forward(x)
        if pred >= 0.5:
            return 1
        else:
            return 0

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
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
    data = np.genfromtxt('/home/adam/dev/ganymede/mlframework/two_circles.txt', dtype=np.float64, comments='#')
    x, y = data[:, [0,1]], data[:, 2]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)
    x_train = torch.from_numpy(x_train).type(torch.float)
    x_test  = torch.from_numpy(x_test).type(torch.float)
    y_train = torch.from_numpy(y_train).type(torch.float)
    y_test  = torch.from_numpy(y_test).type(torch.float)
 
    model = NN(2, 10, 1)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 1000
    losses = []
    for i in range(epochs):
        y_pred = model.forward(x_train).squeeze()
        loss = loss_fn(y_pred, y_train)
        print(f"epoch: {i}, loss: {loss.item()}")
        
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    # Plot decision boundaries for training and test sets
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, x_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, x_test, y_test)
    plt.show()

if __name__ == '__main__':
    main()
