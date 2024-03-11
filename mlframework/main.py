#!/usr/bin/python3

import torch
from torch import Tensor
from torch import nn
from torch.nn import functional as F

class NN(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.fc1 = nn.Linear(10, 4)
        self.fc2 = nn.Linear(4, 2)
    
    def forward(self, x: 'Tensor') -> 'Tensor':
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def main() -> None:
    model = NN()

    x = torch.randn(1000, 10)
    y = torch.randn(1000, 2)

    loss_fn = nn.SoftMarginLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    model.train()
    for t in range(100000):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        if t % 100 == 0:
            print(t, loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(model(x))
    print(y)

if __name__ == '__main__':
    main()
