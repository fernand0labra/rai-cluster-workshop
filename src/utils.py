'''
MIT License

Copyright (c) [2025] [Fernando Labra Caso]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset

'''
Common Pytorch Utilities (Dataset and Network Definition)
'''


def data():
    size = 1000
    f = lambda x: x**3 + 13*x**2 + 21*x + 5
    x = torch.linspace(0, 1, steps=size, dtype=torch.float32)
    y = torch.tensor(list(map(f, x)))
    return x, y


class CustomDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.__len__()

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class CustomNet(nn.Module):

    def __init__(self):
        super(CustomNet, self).__init__()
        self.fc1 = nn.Linear(1, 4)
        self.fc2 = nn.Linear(4, 8)
        self.fc3 = nn.Linear(8, 16)
        self.fc4 = nn.Linear(16, 8)
        self.fc5 = nn.Linear(8, 4)
        self.fc6 = nn.Linear(4, 1)

    def forward(self, input):
        f1 = F.relu(self.fc1(input))
        f2 = F.relu(self.fc2(f1))
        f3 = F.relu(self.fc3(f2))
        f4 = F.relu(self.fc4(f3))
        f5 = F.relu(self.fc5(f4))
        return self.fc6(f5)
