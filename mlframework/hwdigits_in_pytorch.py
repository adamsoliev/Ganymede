#!/usr/bin/python3

import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(suppress=True)
from tqdm import trange
import torch
torch.set_printoptions(sci_mode=False)
import torch.nn as nn
import torch.nn.functional as F

def fetch(url):
  import requests, gzip, os, hashlib, numpy
  fp = os.path.join("/tmp", hashlib.md5(url.encode('utf-8')).hexdigest())
  if os.path.isfile(fp):
    with open(fp, "rb") as f:
      dat = f.read()
  else:
    with open(fp, "wb") as f:
      dat = requests.get(url).content
      f.write(dat)
  return numpy.frombuffer(gzip.decompress(dat), dtype=np.uint8).copy()

# model
class NN(torch.nn.Module):
  def __init__(self):
    super(NN, self).__init__()
    self.l1 = nn.Linear(784, 128, bias=False)
    self.l2 = nn.Linear(128, 10, bias=False)
    self.sm = nn.LogSoftmax(dim=1)
  def forward(self, x):
    x = F.relu(self.l1(x))
    x = self.l2(x)
    x = self.sm(x)
    return x


def main():
    # dataset
    X_train = fetch("http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_train = fetch("http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz")[8:]
    X_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz")[0x10:].reshape((-1, 28, 28))
    Y_test = fetch("http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz")[8:]

    # training
    model = NN()
    loss_function = nn.NLLLoss(reduction='none')
    optim = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0)
    BS = 128
    losses, accuracies = [], []
    for i in (t := trange(1000)):
        samp = np.random.randint(0, X_train.shape[0], size=(BS))
        X = torch.tensor(X_train[samp].reshape((-1, 28*28))).float()
        Y = torch.tensor(Y_train[samp]).long()

        # forward pass
        out = model(X)

        # zero grad
        model.zero_grad()

        # loss
        cat = torch.argmax(out, dim=1)
        accuracy = (cat == Y).float().mean()
        loss = loss_function(out, Y)
        loss = loss.mean()

        # backward pass
        loss.backward()

        # update
        optim.step()

        # debug
        loss, accuracy = loss.item(), accuracy.item()
        losses.append(loss)
        accuracies.append(accuracy)
        t.set_description("loss %.2f accuracy %.2f" % (loss, accuracy))
    plt.ylim(-0.1, 1.1)
    plt.plot(losses)
    plt.plot(accuracies)
    plt.figure()

    # evaluation
    Y_test_preds = torch.argmax(model(torch.tensor(X_test.reshape((-1, 28*28))).float()), dim=1).numpy()
    (Y_test == Y_test_preds).mean()

    # compute gradients in torch
    samp = [0,1,2,3]
    model.zero_grad()
    out = model(torch.tensor(X_test[samp].reshape((-1, 28*28))).float())
    out.retain_grad()
    loss = loss_function(out, torch.tensor(Y_test[samp]).long()).mean()
    loss.retain_grad()
    loss.backward()
    plt.imshow(model.l1.weight.grad)
    plt.figure()
    plt.imshow(model.l2.weight.grad)

    plt.show()

if __name__ == '__main__':
    main()