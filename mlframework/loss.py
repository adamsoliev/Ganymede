
from tensor import Tensor
import numpy as np
from typing import cast

def mse_loss(a_true: 'Tensor', b_pred: 'Tensor') -> 'Tensor':
    l = (a_true.data - b_pred.data) ** 2
    result = Tensor(np.mean(l), {a_true, b_pred}, "mse_loss")
    def _backward() -> None:
        n = a_true.data.shape[0]
        b_pred.grad += -2 * (a_true.data - b_pred.data) / n
        a_true.grad += -2 * (b_pred.data - a_true.data) / n
    result._backward = _backward
    return result