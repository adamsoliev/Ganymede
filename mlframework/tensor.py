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
        # return f"tensor.size([{len(self.data)}, {len(self.data[0])}])"
        if self.dim == 1:
            return [len(self.data)]
        elif self.dim == 2:
            return [len(self.data), len(self.data[0])]
    
    def matmul(self, other):
        row1, col1 = self.size()
        row2, col2 = other.size()
        if (row1 != col2 or row2 != col1): assert 0

        result = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
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


# ------------- ctorch TEST -------------
def sanitycheck():
    in1 = [[0.1, 1.2], [2.2, 3.1], [4.9, 5.2]]
    in2 = [[9.2, 2.3, 7.9], [4.6, 2.7, 7.3]]

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
    
# ------------- MAIN -------------
def main():
    sanitycheck()

if __name__ == '__main__':
    main()