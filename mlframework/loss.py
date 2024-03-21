
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

def binary_cross_entropy(y_pred: 'Tensor', y_true: 'Tensor') -> 'Tensor':
    eps = 1e-12
    y_pred.data = np.clip(y_pred.data, eps, 1 - eps)

    loss = - (y_true.data * np.log(y_pred.data) + (1 - y_true.data) * np.log(1 - y_pred.data))
    result = Tensor(np.mean(loss), {y_pred, y_true}, "bce_loss")
    def _backward() -> None:
        y_pred.grad += - (y_true.data / y_pred.data) + (1 - y_true.data) / (1 - y_pred.data)
        # y_true.grad += (y_pred.data - y_true.data) / (y_pred.data * (1 - y_pred.data))  
    result._backward = _backward
    return result