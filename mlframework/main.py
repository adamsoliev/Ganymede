#!/usr/bin/python3

# src https://github.com/bstollnitz/fashion-mnist-pytorch

"""
1. Pytorch examples
    Pytorch primer
2. Implement something yourself
3. Repeat with NumPy
    Pytorch NumPy

if interested in machine learning, the scikit-learn docs actually make for great notes
"""


"""Training, evaluation, and prediction."""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn.modules.loss import CrossEntropyLoss
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

from neural_network import NeuralNetwork

labels_map = {
    0: 'T-Shirt',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot',
}

DATA_DIRPATH = 'fashion-mnist-pytorch/data'
MODEL_DIRPATH = 'fashion-mnist-pytorch/model'
IMAGE_FILEPATH = 'fashion-mnist-pytorch/predict-image.png'


def _get_data(batch_size: int) -> Tuple[DataLoader, DataLoader]:
    """Downloads Fashion MNIST data, and returns two DataLoader objects
    wrapping test and training data."""
    training_data = datasets.FashionMNIST(
        root=DATA_DIRPATH,
        train=True,
        download=True,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root=DATA_DIRPATH,
        train=False,
        download=True,
        transform=ToTensor(),
    )

    train_dataloader = DataLoader(training_data,
                                  batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return (train_dataloader, test_dataloader)


def _fit(device: str, dataloader: DataLoader, model: nn.Module,
         loss_fn: CrossEntropyLoss,
         optimizer: Optimizer) -> Tuple[float, float]:
    """Trains the given model for a single epoch."""
    loss_sum = 0
    correct_item_count = 0
    item_count = 0

    # Used for printing only.
    batch_count = len(dataloader)
    print_every = 100

    model.to(device)
    model.train()

    for batch_index, (x, y) in enumerate(dataloader):
        x = x.float().to(device)
        y = y.long().to(device)

        (y_prime, loss) = _fit_one_batch(x, y, model, loss_fn, optimizer)

        correct_item_count += (y_prime.argmax(1) == y).sum().item()
        loss_sum += loss.item()
        item_count += len(x)

        # Printing progress.
        if ((batch_index + 1) % print_every == 0) or ((batch_index + 1)
                                                      == batch_count):
            accuracy = correct_item_count / item_count
            average_loss = loss_sum / item_count
            print(f'[Batch {batch_index + 1:>3d} - {item_count:>5d} items] ' +
                  f'loss: {average_loss:>7f}, ' +
                  f'accuracy: {accuracy*100:>0.1f}%')

    average_loss = loss_sum / item_count
    accuracy = correct_item_count / item_count

    return (average_loss, accuracy)


def _fit_one_batch(x: torch.Tensor, y: torch.Tensor, model: NeuralNetwork,
                   loss_fn: CrossEntropyLoss,
                   optimizer: Optimizer) -> Tuple[torch.Tensor, torch.Tensor]:
    """Trains a single minibatch (backpropagation algorithm)."""
    y_prime = model(x)
    loss = loss_fn(y_prime, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return (y_prime, loss)


def _evaluate(device: str, dataloader: DataLoader, model: nn.Module,
              loss_fn: CrossEntropyLoss) -> Tuple[float, float]:
    """Evaluates the given model for the whole dataset once."""
    loss_sum = 0
    correct_item_count = 0
    item_count = 0

    model.to(device)
    model.eval()

    with torch.no_grad():
        for (x, y) in dataloader:
            x = x.float().to(device)
            y = y.long().to(device)

            (y_prime, loss) = _evaluate_one_batch(x, y, model, loss_fn)

            correct_item_count += (y_prime.argmax(1) == y).sum().item()
            loss_sum += loss.item()
            item_count += len(x)

        average_loss = loss_sum / item_count
        accuracy = correct_item_count / item_count

    return (average_loss, accuracy)


def _evaluate_one_batch(
        x: torch.tensor, y: torch.tensor, model: NeuralNetwork,
        loss_fn: CrossEntropyLoss) -> Tuple[torch.Tensor, torch.Tensor]:
    """Evaluates a single minibatch."""
    with torch.no_grad():
        y_prime = model(x)
        loss = loss_fn(y_prime, y)

    return (y_prime, loss)


def training_phase(device: str):
    """Trains the model for a number of epochs, and saves it."""
    learning_rate = 0.1
    batch_size = 64
    epochs = 5

    (train_dataloader, test_dataloader) = _get_data(batch_size)

    model = NeuralNetwork()

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    print('\n***Training***')
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1}\n-------------------------------')
        (train_loss, train_accuracy) = _fit(device, train_dataloader, model,
                                            loss_fn, optimizer)
        print(f'Train loss: {train_loss:>8f}, ' +
              f'train accuracy: {train_accuracy * 100:>0.1f}%')

    print('\n***Evaluating***')
    (test_loss, test_accuracy) = _evaluate(device, test_dataloader, model,
                                           loss_fn)
    print(f'Test loss: {test_loss:>8f}, ' +
          f'test accuracy: {test_accuracy * 100:>0.1f}%')

    Path(MODEL_DIRPATH).mkdir(exist_ok=True)
    path = Path(MODEL_DIRPATH, 'weights.pth')
    torch.save(model.state_dict(), path)


def _predict(model: nn.Module, x: torch.Tensor, device: str) -> np.ndarray:
    """Makes a prediction for input x."""
    model.to(device)
    model.eval()

    x = torch.from_numpy(x).float().to(device)

    with torch.no_grad():
        y_prime = model(x)
        probabilities = nn.functional.softmax(y_prime, dim=1)
        predicted_indices = probabilities.argmax(1)
    return predicted_indices.cpu().numpy()


def inference_phase(device: str):
    """Makes a prediction for a local image."""
    print('\n***Predicting***')

    model = NeuralNetwork()
    path = Path(MODEL_DIRPATH, 'weights.pth')
    model.load_state_dict(torch.load(path))

    with Image.open(IMAGE_FILEPATH) as image:
        x = np.asarray(image).reshape((-1, 28, 28)) / 255.0

    predicted_index = _predict(model, x, device)[0]
    predicted_class = labels_map[predicted_index]

    print(f'Predicted class: {predicted_class}')


def main() -> None:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    training_phase(device)
    inference_phase(device)


if __name__ == '__main__':
    main()
    # W = torch.tensor([[1., 2.]], requires_grad=True)
    # X = torch.tensor([[3.], [4.]])
    # b = torch.tensor([5.], requires_grad=True)
    # y = torch.tensor([[6.]]) # true value
    # y_prime = torch.matmul(W, X) + b
    # loss_fn = torch.nn.MSELoss()
    # loss = loss_fn(y_prime, y)
    # loss.backward()
    # print("W: ", W.size(), W.grad)
    # print("b: ", b.size(), b.grad)
