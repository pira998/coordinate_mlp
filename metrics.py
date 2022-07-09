# pytorch lightning metric PSNR implementation
import torch
from torch import nn

@torch.no_grad()
def PSNR(x, y):
    return 10 * torch.log10(1 / torch.mean((x - y) ** 2))