#!/usr/bin/python3

# adapted from https://www.learnpytorch.io/02_pytorch_classification/

import torch
from torch import Tensor
import numpy as np
from torch import nn
from matplotlib import pyplot as plt    # type: ignore
from sklearn import datasets            # type: ignore
from sklearn.model_selection import train_test_split

# Hyperparameters
HL = 20
EPOCHS = 3000
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

class NN(nn.Module):
    def __init__(self, input_size: int, H1: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(input_size, H1)
        self.sigmoid1 = nn.Sigmoid()
        self.linear2 = nn.Linear(H1, output_size)
        self.sigmoid2 = nn.Sigmoid()
    
    def forward(self, x: 'Tensor') -> 'Tensor':
        x = self.linear(x)
        x = self.sigmoid1(x)
        x = self.linear2(x)
        x = self.sigmoid2(x)
        return x

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred)) * 100 
    return acc

def main() -> None:
    # create circles
    n_pts = 1000
    # (1000, 2) (1000,)
    X, y = datasets.make_circles(n_samples=n_pts, random_state=42, noise=0.04)

    # plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    # plt.show()

    # create tensors
    X = torch.from_numpy(X).type(torch.float)
    y = torch.from_numpy(y).type(torch.float)

    # # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42) # make the random split reproducible

    # usual stuff
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NN(2, HL, 1).to(device)
    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    epochs = EPOCHS

    # move data to device
    X_train, y_train = X_train.to(device), y_train.to(device)
    X_test, y_test = X_test.to(device), y_test.to(device)

    for epoch in range(epochs):
        model.train()

        y_prob_distr = model(X_train).squeeze()
        loss = loss_fn(y_prob_distr, y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=torch.round(y_prob_distr))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_prob_distr = model(X_test).squeeze()
            # 2. Caculate loss/accuracy
            test_loss = loss_fn(test_prob_distr, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=torch.round(test_prob_distr))

        # Print out what's happening every 10 epochs
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
    
    assert test_acc > 90.0

    # visualize 
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title("Train")
    plot_decision_boundary(model, X_train, y_train)
    plt.subplot(1, 2, 2)
    plt.title("Test")
    plot_decision_boundary(model, X_test, y_test)
    plt.show()

if __name__ == '__main__':
    main()
