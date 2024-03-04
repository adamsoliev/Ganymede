#!/usr/bin/python3

# TODO: Manipulating tensor shapes 

# %mathplotlib inline

import torch
# import matplotlib.pyplot as plt
# import matplotlib.ticker as ticker
# import math

BATCH_SIZE = 16
DIM_IN = 1000
HIDDEN_SIZE = 100
DIM_OUT = 10

class TinyModel(torch.nn.Module):

    def __init__(self):
        super(TinyModel, self).__init__()

        self.layer1 = torch.nn.Linear(DIM_IN, HIDDEN_SIZE)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(HIDDEN_SIZE, DIM_OUT)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu(x)
        x = self.layer2(x)
        return x

def main():
    some_input = torch.randn(BATCH_SIZE, DIM_IN, requires_grad=False)
    ideal_output = torch.randn(BATCH_SIZE, DIM_OUT, requires_grad=False)

    model = TinyModel()
    print(model.layer2.weight[0][0:10]) # just a small slice
    print(model.layer2.weight.grad)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

    prediction = model(some_input)

    loss = (ideal_output - prediction).pow(2).sum()
    print(loss)

    loss.backward()
    print(model.layer2.weight[0][0:10])
    print(model.layer2.weight.grad[0][0:10])

    optimizer.step()
    print(model.layer2.weight[0][0:10])
    print(model.layer2.weight.grad[0][0:10])

if __name__ == '__main__':
    main()
