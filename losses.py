import torch
from torch import nn

class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()

    def forward(self, x, y):
        return torch.mean((x - y) ** 2)