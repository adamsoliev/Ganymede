#!/usr/bin/python3

import torch
from torch import nn
from torch.nn import functional as F

class NN(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(10, 4)
        self.fc2 = nn.Linear(4, 2)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def main():
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

# def main():
#     learning_rate = 0.1
#     batch_size = 64
#     epochs = 5

#     model = NeuralNetwork()
#     loss_fn = CrossEntropyLoss()
#     optimizer = SGW(model.parameters(), lr=learning_rate)

#     for epoch in range(epochs):
#         print(f'\nEpoch {epoch + 1}\n-------------------------------')

#         loss_sum = 0
#         correct_item_count = 0
#         item_count = 0
#         for batch_index, (x, y) in enumerate(data):
#             # x - input; y - true_value
#             y_prime = model(x)
#             loss = loss_fn(y_prime, y)
#             optimizer.zero_grad()
#             loss.backward()
#             optimizer.step()

#             correct_item_count += (y_prime.argmax(1) == y).sum().item()
#             loss_sum += loss.item()
#             item_count += len(x)

#         average_loss = loss_sum / item_count
#         accuracy = correct_item_count / item_count
#         print(f'loss: {average_loss:>8f}, ' + f'accuracy: {accuracy * 100:>0.1f}%')


