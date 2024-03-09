import unittest

from tensor import Tensor

class TestTensorSum(unittest.TestCase):
    def test_simple_add(self):
        t1 = Tensor([1., 2., 3.])
        t2 = Tensor([4., 5., 6.])
        t3 = t1 + t2

        assert t3.data.tolist() == [5., 7., 9.]

        t3.backward()

        assert t1.grad.data.tolist() == [1., 1., 1.]
        assert t2.grad.data.tolist() == [1., 1., 1.]
        assert t3.grad.data.tolist() == [1., 1., 1.]
    
    # Broadcasting
    #   1. assume missing dimentions have size one
    #   2. treat dimentions with size one expandable
    def test_broadcast_add(self):
        t1 = Tensor([[3., 4., 5.], [4., 3., 6.]])
        t2 = Tensor([9., 4., 1.])
        t3 = t1 + t2

        assert t3.data.tolist() == [[12.,  8.,  6.], [13.,  7.,  7.]]

        t3.backward()

        assert t1.grad.data.tolist() == [[1., 1., 1.], [1., 1., 1.]]
        assert t2.grad.data.tolist() == [2., 2., 2.]
        assert t3.grad.data.tolist() == [[1., 1., 1.], [1., 1., 1.]]