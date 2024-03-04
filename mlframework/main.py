#!/usr/bin/python3

# TODO: Manipulating tensor shapes 

import torch

def main():
    lin = torch.nn.Linear(3,2) # 3 inputs and 2 output
    # print(lin)
    # print(lin.weight[0])
    # print(lin.bias[0])
    x = torch.rand(1,3)
    y = lin(x) # forward pass

    print(x[0].matmul(lin.weight[0]).sum().add(lin.bias[0]))
    print(x[0].matmul(lin.weight[1]).sum().add(lin.bias[1]))

    print(y)

    """
    nn.Linear(3, 2) can as well be interpreted as 3 x 2 matrix of weights and vector of length 2 of biases

    weights =           # input x output matrix
    [[w01, w02, w03]    
     [w11, w12, w13]]

    biases =            # for each output
    [b1, b2]

    when you run forward pass
    y = lin(x)
        under the hood, this will happen
            o1 = x1 * w01 + x2 * w02 + x3 * w03 + b1 
            o2 = x1 * w11 + x2 * w12 + x3 * w13 + b2 
        y = [o1, o2]
    """

    # # below is mimicking the above with exact numbers
    # input = torch.tensor([[0.8790, 0.9774, 0.2547]])
    # weights = torch.tensor([[0.1656, 0.4969, -0.4972], [-0.2035, -0.2579, -0.3780]])
    # biases = torch.tensor([0.3768, 0.3781])

    # print(input[0].matmul(weights[0]).sum().add(biases[0])) 
    # print(input[0].matmul(weights[1]).sum().add(biases[1])) 

if __name__ == '__main__':
    main()
