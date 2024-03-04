#!/usr/bin/python3

# TODO: Manipulating tensor shapes 

import torch
import torch.functional as F

class LeNet(torch.nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()
        # kernel
            # 1 - # of input channels (black & white)
            # 6 - # of features (output channels)
            # 5 - kernel size (5x5)

            # it outputs tensor of 6x28x28
                # 6 - # of features
                # 28x28 - height and width (scanning 32-pixel row by 5x5 kernel gives you 28 positions)
        self.conv1 = torch.nn.Conv2d(1, 6, 5) 
        self.conv2 = torch.nn.Conv2d(6, 16, 3)
        self.fc1 = torch.nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a 2x2 window (basically in this line: 6x32x32 -1> 6x28x28 -2> 6x14x14 due to 1 - conv; 2 - maxpool)
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # Max pooling over a 2x2 window (basically in this line: 6x14x14 -> 6x12x12 -> 16x6x6)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # 16x6x6 -> 576-element vector (reshape before feeding into fully connected layer)
        x = x.view(-1, self.num_flat_features(x)) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def main():
    nn = LeNet()
    print(nn)

if __name__ == '__main__':
    main()
