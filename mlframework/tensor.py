#!/usr/bin/python3
import torch

class Tensor:
    def __init__(self, data, dim = 2):
        self.data = data
        self.dim = dim

    def __repr__(self):
        formatted = "tensor([\n"
        if self.dim == 1: # 1d
            formatted += "["
            for val in self.data:
                formatted += f"{val:.4f}, "
            formatted += "]"
        elif self.dim == 2: # 2d
            for row in self.data:
                formatted += "        [" + ", ".join(f"{val:.4f}" for val in row) + "],\n"
        else:
            assert 0
        formatted += "])"
        return formatted
    
    def size(self):
        if self.dim == 1:
            return [len(self.data)]
        elif self.dim == 2:
            return [len(self.data), len(self.data[0])]

        """
        Consider you have two matrices A and B of orders a1xa2 and b1xb2 respectively.

        Matrix addition/subtraction on the two matrices will be defined iff a1=b1 and a2=b2
        Matrix multiplication on them is defined iff a2=b1 for AB to be defined and 
        b2=a1 for BA to be defined. AB will be of order a1*b2 and BA will be of order b1*a2
        """
    def add(self, other):
        assert isinstance(other, Tensor)

        if self.dim == 1:
            assert 0

        assert self.dim == 2
        row1, col1 = self.size()
        row2, col2 = other.size()
        if row1 != row2 or col1 != col2: assert 0
        result = [[0] * row1 for _ in range(col2)]
        for i in range(row1):
            for k in range(col1):
                result[i][k] = self.data[i][k] + other.data[i][k]
        return Tensor(result)

    def sub(self, other):
        assert isinstance(other, Tensor)

        if self.dim == 1:
            assert 0

        assert self.dim == 2
        row1, col1 = self.size()
        row2, col2 = other.size()
        if row1 != row2 or col1 != col2: assert 0
        result = [[0] * row1 for _ in range(col2)]
        for i in range(row1):
            for k in range(col1):
                result[i][k] = self.data[i][k] - other.data[i][k]
        return Tensor(result)

    def equal(self, other):
        assert isinstance(other, Tensor)

        if self.dim == 1:
            if (len(self.data) != len(other.data)): return False
            for i in range(len(self.data)):
                if self.data[i] != other.data[i]: return False

        assert self.dim == 2
        row1, col1 = self.size()
        row2, col2 = other.size()
        if row1 != row2 or col1 != col2: return False
        for i in range(row1):
            for k in range(col1):
                if self.data[i][k] != other.data[i][k]: return False
        return True
    
    def mul(self, other):
        assert isinstance(other, (int, float))
        if self.dim == 1:
            pass
        assert self.dim == 2
        row1, col1 = self.size()
        result = [[0] * col1 for _ in range(row1)]
        for i in range(row1):
            for k in range(col1):
               result[i][k] = self.data[i][k] * other
        return Tensor(result)
    
    def matmul(self, other):
        assert isinstance(other, Tensor)

        if self.dim == 1:
            assert 0

        assert self.dim == 2
        row1, col1 = self.size()
        row2, col2 = other.size()
        if (row1 != col2 or row2 != col1): 
            print(f"matrix dimentions mismatch: ({row1, col1}) vs ({row2, col2})")
            assert 0

        result = [[0] * row1 for _ in range(col2)]
        for i in range(row1):
            for j in range(col2):
                for k in range(col1):
                    result[i][j] += self.data[i][k] * other.data[k][j]
        return Tensor(result)
    
    def transpose(self):
        if self.dim == 1:
            assert 0

        assert self.dim == 2
        row1, col1 = self.size()
        result = [[0] * row1 for _ in range(col1)]
        for i in range(row1):
            for k in range(col1):
                result[k][i] = self.data[i][k]
        return Tensor(result)

# ------------- ctorch API -------------
def tensor(data):
    return Tensor(data)

def matmul(t1, t2):
    return t1.matmul(t2)

# return a new copy
def flatten(t1):
    flattened = []
    row1, col1 = t1.size()
    for i in range(row1):
        for j in range(col1):
            flattened.append(t1.data[i][j])
    return Tensor(flattened, 1)

def equal(t1, t2):
    return t1.equal(t2)

# ------------- ctorch TEST -------------
lom = [
    [[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]],
    [[9.2, 2.3, 7.9], [4.6, 2.7, 7.3]],
    [[2.2, 3.1], [4.9, 5.2]],
    [[9.2, 2.3], [4.6, 2.7]],
    [[2.2, 3.1], [4.9, 5.2]],
    [[2.2, 3.1], [4.9, 5.2]],
    [[0.0, 0.0], [0.0, 0.0]],
    [[0.0, 0.0], [0.0, 0.0]],
    [[-2.2, 3.1], [4.9, 5.2]],
    [[2.2, 3.1], [-4.9, 5.2]]
]

def test_matmul():
    def cassert(in1, in2):
        # pytorch
        a = torch.tensor(in1)
        b = torch.tensor(in2)
        c = torch.matmul(a, b)
        f = torch.flatten(c)

        # ours
        t1 = tensor(in1)
        t2 = tensor(in2)
        c1 = matmul(t1, t2)
        f1 = flatten(c1)

        for i, val in enumerate(f):
            assert round(val.item(), 4) == round(f1.data[i], 4)
    
    for i in range(0, len(lom), 2):
        cassert(lom[i], lom[i + 1])

def test_add():
    def cassert(in1, in2):
        # --
        a = torch.tensor(in1)
        b = torch.tensor(in2)
        r = a.add(b)
        f = torch.flatten(r)
        # --
        t1 = tensor(in1)
        t2 = tensor(in2)
        r1 = t1.add(t2)
        f1 = flatten(r1)

        for i, val in enumerate(f):
            assert round(val.item(), 4) == round(f1.data[i], 4)

    for i in range(2, len(lom), 2):
        cassert(lom[i], lom[i + 1])

def test_sub():
    def cassert(in1, in2):
        # --
        a = torch.tensor(in1)
        b = torch.tensor(in2)
        r = a.sub(b)
        f = torch.flatten(r)
        # --
        t1 = tensor(in1)
        t2 = tensor(in2)
        r1 = t1.sub(t2)
        f1 = flatten(r1)

        for i, val in enumerate(f):
            assert round(val.item(), 4) == round(f1.data[i], 4)

    for i in range(2, len(lom), 2):
        cassert(lom[i], lom[i + 1])

def test_mul():
    def cassert(in1, num):
        # --
        a = torch.tensor(in1)
        r = a.mul(num)
        # --
        t1 = tensor(in1)
        r1 = t1.mul(num)

        for i, fv in enumerate(r):
            for k, val in enumerate(fv):
                assert round(val.item(), 4) == round(r1.data[i][k], 4)

    in1 = [[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]
    cassert(in1, 2.5)

    in2 = [[0.1, 2.2, 4.9], [1.2, 3.1, 5.2]]
    cassert(in2, -1.2)

    in3 = [[0.0, 2.2, 4.9], [-1.2, 3.1, 5.2]]
    cassert(in3, 10.3)

def test_equal():
    def cassert(in1, in2):
        # --
        a = torch.tensor(in1)
        b = torch.tensor(in2)
        r = a.equal(b)
        # --
        t1 = tensor(in1)
        t2 = tensor(in2)
        r1 = t1.equal(t2)
        assert r == r1

    for i in range(0, len(lom), 2):
        cassert(lom[i], lom[i + 1])

def test_transpose():
    def cassert(in1, tin1):
        t1 = tensor(in1)
        t2 = tensor(tin1)
        r1 = t1.transpose()
        assert r1.equal(t2)

    in1 = [[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]
    tin1 = [[0.1, 2.2, 4.9], [1.2, 3.1, 5.2]]
    cassert(in1, tin1)

    in1 = [[9.2, 2.3, 7.9], [4.6, 2.7, 7.3]]
    in2 = [[9.2, 4.6], [2.3, 2.7], [7.9, 7.3]]
    cassert(in1, in2)

    
# ------------- MAIN -------------
def main():
    test_add()
    test_sub()
    test_mul()
    test_equal()
    test_matmul()
    test_transpose()

if __name__ == '__main__':
    main()