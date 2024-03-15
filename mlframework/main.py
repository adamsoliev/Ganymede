#!/usr/bin/python3

import torch
import numpy as np
from torch import Tensor
from torch import nn
from torch.nn import functional as F
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split

class NN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.layer1 = nn.Linear(2, 5)
        self.layer2 = nn.Linear(5, 1)
    
    def forward(self, x: 'Tensor') -> 'Tensor':
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return x

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item() 
    acc = (correct / len(y_pred)) * 100 
    return acc

def plot_decision_boundary(model: torch.nn.Module, X: torch.Tensor, y: torch.Tensor):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    # Put everything to CPU (works better with NumPy + Matplotlib)
    model.to("cpu")

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = torch.from_numpy(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    model.eval()
    with torch.inference_mode():
        y_logits = model(X_to_pred_on)

    # Test for multi-class or binary and adjust logits to prediction labels
    # if len(torch.unique(y)) > 2:
    #     y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  # mutli-class
    # else:
    y_pred = torch.round(y_logits)  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())


# Plot linear data or training and test and predictions (optional)
def plot_predictions( train_data, train_labels, test_data, test_labels, predictions=None):
    """ Plots linear training data and test data and compares predictions.  """
    plt.figure(figsize=(10, 7))

    # Plot training data in blue
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        # Plot the predictions in red (predictions were made on the test data)
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

    # Show the legend
    plt.legend(prop={"size": 14})


def main() -> None:
    # model = NN()

    # x = torch.randn(10)
    # y = torch.randn(10)

    # plt.scatter(x, y)
    # plt.show()

    # loss_fn = nn.SoftMarginLoss()
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)

    # model.train()
    # for t in range(100000):
    #     y_pred = model(x)
    #     loss = loss_fn(y_pred, y)

    #     if t % 100 == 0:
    #         print(t, loss.item())

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()
    
    # print(model(x))
    # print(y)

    device = "cude" if torch.cuda.is_available() else "cpu"

    ### DATA
    data = np.genfromtxt('two_circles.txt', dtype=np.float64, comments='#')
    x, y = data[:, [0,1]], data[:, 2]
    # plt.scatter(x=x[:, 0], y=x[:, 1], c=y, cmap=plt.cm.RdYlBu)
    # plt.show()

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    x_train = torch.from_numpy(x_train).type(torch.float).to(device) 
    x_test  = torch.from_numpy(x_test).type(torch.float).to(device)
    y_train = torch.from_numpy(y_train).type(torch.float).to(device)
    y_test  = torch.from_numpy(y_test).type(torch.float).to(device)
 
    model = NN()
    # y_prob = model(x_train)
    # y_pred = torch.round(y_prob)
    # y_pred_label = torch.round(model(x_test))
    # print(y_pred.shape)
    # print(y_pred_label.shape)

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.1)

    torch.manual_seed(42)

    epochs = 100
    for epoch in range(epochs):
        ### Training
        model.train()

        # 1. Forward pass (model outputs raw logits)
        y_prob = model(x_train) 
        y_pred = torch.round(y_prob) 
        y_pred = y_pred.squeeze() # remove extra dimention
    
        assert y_pred.shape == y_train.shape
        loss = loss_fn(y_pred, y_train) 
        acc = accuracy_fn(y_true=y_train, y_pred=y_pred) 

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        ### Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_prob = model(x_test)
            test_pred = torch.round(test_prob)
            test_pred = test_pred.squeeze()
            # 2. Caculate loss/accuracy
            assert test_pred.shape == y_test.shape
            test_loss = loss_fn(test_pred, y_test)
            test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)

        # Print out what's happening every 10 epochs
        if epoch % 10 == 0:
            print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
    
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
