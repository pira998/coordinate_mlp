import torch 
from torch import nn 



class MLP(nn.Module):
    '''
    MLP with 4 hidden layers
    '''
    def __init__(self, n_layers=4, n_hidden=256, n_input=2, n_output=3):
        super(MLP, self).__init__()
        self.n_layers = n_layers
        self.n_hidden = n_hidden
        self.n_input = n_input
        self.n_output = n_output
        self.layers = [nn.Linear(self.n_input, self.n_hidden), nn.ReLU(True)]
        for i in range(self.n_layers - 1):
            if i == self.n_layers - 2:
                self.layers.append(nn.Linear(self.n_hidden, self.n_hidden))
                self.layers.append(nn.ReLU(True))
            else:
                self.layers.append(nn.Linear(self.n_hidden, self.n_output))
                self.layers.append(nn.Sigmoid())
        # self.net = nn.ModuleList(self.layers)
        self.net = nn.Sequential(*self.layers)

        def forward(self, x):
            """
            x: (B, 2) # pixel uv(normalized)
            """
            return self.net(x) # (B, 3) rgb




class PE(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(PE, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x