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
    
    def equal(self, other):
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
    
    def matmul(self, other):
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
    
# ------------- MAIN -------------
def main():
    test_matmul()
    test_equal()

if __name__ == '__main__':
    main()