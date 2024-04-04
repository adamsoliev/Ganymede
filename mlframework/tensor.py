# inspired by https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py

import numpy as np
from tinygrad import Tensor as __Tensor # type:ignore
from tinygrad import dtypes # type:ignore

# Today's goal
# input = np.random.randn(3, 2)
# target = np.random.rand(3, 2)
# a = Tensor(input, dtype=dtypes.float32)
# b = Tensor(target, dtype=dtypes.float32)

# c = a.sigmoid()
# loss = c.binary_crossentropy(b)
# loss.backward()

# print(loss.numpy())

class Tensor():
    def __init__(self, data):
        self.data = data

    def __repr__(self): return f"Tensor {self.data}"

    def sigmoid(self):
        return Tensor(1 / (1 + np.exp(-self.data)))
    
    def numpy(self):
        return self.data

    def binary_crossentropy(self, y):
        return Tensor(np.mean(-y.numpy() * np.log(self.data) - (1 - y.numpy()) * np.log(1 - self.data)))


def main():
    np.random.seed(23)
    input = np.random.randn(3, 2)
    target = np.random.rand(3, 2)

    a = __Tensor(input, dtype=dtypes.float32)
    b = __Tensor(target, dtype=dtypes.float32)
    c = a.sigmoid()
    loss = c.binary_crossentropy(b)

    _a = Tensor(input)
    _b = Tensor(target)
    _c = _a.sigmoid()
    _loss = _c.binary_crossentropy(_b)

    assert np.allclose(c.numpy(), _c.numpy())
    assert np.allclose(loss.numpy(), _loss.numpy())

if __name__ == "__main__":
    main()