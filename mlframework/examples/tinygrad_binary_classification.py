#!/usr/bin/python3

# adapted from https://www.learnpytorch.io/02_pytorch_classification/

import tinygrad
from tinygrad import Tensor, dtypes
from tinygrad.nn.optim import SGD, Adam
import numpy as np
from matplotlib import pyplot as plt    # type: ignore
from sklearn import datasets            # type: ignore
from sklearn.model_selection import train_test_split

# Hyperparameters
HL = 20
EPOCHS = 200
LR = 0.001

def plot_decision_boundary(model, X, y):
    """Plots decision boundaries of model predicting on X in comparison to y.

    Source - https://madewithml.com/courses/foundations/neural-networks/ (with modifications)
    """
    X = X.numpy(); y = y.numpy()

    # Setup prediction boundaries and grid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 101), np.linspace(y_min, y_max, 101))

    # Make features
    X_to_pred_on = Tensor(np.column_stack((xx.ravel(), yy.ravel()))).float()

    # Make predictions
    y_prob_distr = model.forward(X_to_pred_on)
    y_pred = Tensor.round(y_prob_distr)  # binary

    # Reshape preds and plot
    y_pred = y_pred.reshape(xx.shape).detach().numpy()
    plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.RdYlBu)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())

class Linear:
    def __init__(self, in_features, out_features, bias=True, initialization: str='kaiming_uniform'):
        self.weight = getattr(Tensor, initialization)(out_features, in_features)
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x):
        return x.linear(self.weight.transpose(), self.bias)

    def parameters(self):
        return [self.weight, self.bias]

class NN():
    def __init__(self, input_size: int, H1: int, output_size: int):
        super().__init__()
        self.linear = Linear(input_size, H1)
        self.linear2 = Linear(H1, output_size)
    
    def forward(self, x: 'Tensor') -> 'Tensor':
        x = self.linear(x).sigmoid()
        x = self.linear2(x).sigmoid()
        return x
    
    def parameters(self):
        params = []
        for p in self.linear.parameters():
            params.append(p)
        for p in self.linear2.parameters():
            params.append(p)
        return params

# Calculate accuracy (a classification metric)
def accuracy_fn(y_true, y_pred):
    correct = np.equal(y_true.numpy(), y_pred.numpy()).sum() # torch.eq() calculates where two tensors are equal
    acc = (correct / len(y_pred.numpy())) * 100 
    return acc

def main() -> None:
    # create circles
    n_pts = 1000
    # (1000, 2) (1000,)
    X, y = datasets.make_circles(n_samples=n_pts, random_state=42, noise=0.04)

    # plt.scatter(x=X[:, 0], y=X[:, 1], c=y, cmap=plt.cm.RdYlBu)
    # plt.show()

    # # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                        random_state=42) # make the random split reproducible
    # create tensors
    X_train = Tensor(X_train, dtype=dtypes.float32)
    X_test  = Tensor(X_test , dtype=dtypes.float32)
    y_train = Tensor(y_train, dtype=dtypes.float32)
    y_test  = Tensor(y_test , dtype=dtypes.float32)

    # usual stuff
    model = NN(2, HL, 1)
    optimizer = Adam(model.parameters(), lr=LR)

    epochs = EPOCHS

    for epoch in range(epochs):
        y_prob_distr = model.forward(X_train).squeeze()
        loss = y_prob_distr.binary_crossentropy(y_train)
        acc = accuracy_fn(y_true=y_train, y_pred=Tensor.round(y_prob_distr))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Testing
        # 1. Forward pass
        test_prob_distr = model.forward(X_test).squeeze()
        # 2. Caculate loss/accuracy
        test_loss = test_prob_distr.binary_crossentropy(y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=Tensor.round(test_prob_distr))

        # Print out what's happening every 10 epochs
        if epoch % 100 == 0:
            print(f"Epoch: {epoch} | Loss: {loss.numpy()}, Accuracy: {acc} | Test loss: {test_loss.numpy()}, Test acc: {test_acc}")
    
    # assert test_acc > 90.0

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
