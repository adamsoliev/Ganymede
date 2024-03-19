
from tensor import Tensor
import numpy as np

def mse_loss(a_true: 'Tensor', b_pred: 'Tensor') -> 'Tensor':
    l = (a_true.data - b_pred.data) ** 2
    result = Tensor(np.mean(l), {a_true, b_pred}, "mse_loss")
    def _backward() -> None:
        n = a_true.data.shape[0]
        b_pred.grad += -2 * (a_true.data - b_pred.data) / n
        a_true.grad += -2 * (b_pred.data - a_true.data) / n
    result._backward = _backward
    return result

def bce_loss(a_true: 'Tensor', b_pred: 'Tensor') -> 'Tensor':
    loss = -(a_true.data * np.log(b_pred.data) + (1 - a_true.data) * np.log(1 - b_pred.data)).mean()
    result = Tensor(loss, {a_true, b_pred}, "bce_loss")
    def _backward() -> None:
        b_pred.grad += -np.divide(a_true.data, b_pred.data) + np.divide((1 - a_true.data),(1 - b_pred.data))
        a_true.grad += -(np.log(b_pred.data) - np.log(1 - b_pred.data))
    result._backward = _backward
    return result
