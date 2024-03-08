import unittest

from tensor import Tensor

class TestTensorSum(unittest.TestCase):
    def test_simple_add(self):
        t1 = Tensor([1, 2, 3])
        t2 = Tensor([4, 5, 6])
        t3 = t1 + t2

        assert t3.data.tolist() == [5, 7, 9]

        t3.backward()

        assert t1.grad.data.tolist() == [1, 1, 1]
        assert t2.grad.data.tolist() == [1, 1, 1]
        assert t3.grad.data.tolist() == [1, 1, 1]