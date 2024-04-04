# inspired by https://github.com/tinygrad/tinygrad/blob/master/tinygrad/tensor.py

import numpy as np
from tinygrad import Tensor # type:ignore
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

class _Tensor():
    def __init__(self, data):
        self.data = data

    def __repr__(self): return f"Tensor {self.data}"

    def sigmoid(self):
        return Tensor(1 / (1 + np.exp(-self.data)))
    
    def numpy(self):
        return self.data


def main():
    np.random.seed(23)
    input = np.random.randn(3, 2)

    a = Tensor(input, dtype=dtypes.float32)
    c = a.sigmoid()
    print(c.numpy())

    _a = _Tensor(input)
    _c = _a.sigmoid()
    print(_c.numpy())

if __name__ == "__main__":
    main()