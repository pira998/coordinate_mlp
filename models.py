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
            if i != self.n_layers - 2:
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
    """
    perform positional encoding
    """
    def __init__(self, P):
        """
        P:(2, F) encoding matrix
        """
        super(PE, self).__init__()
        # self.P = P
        self.register_buffer('P', P)
        
    @property
    def out_dim(self):
        return self.P.shape[1] * 2

    def forward(self, x):
        """
        x: (B, 2)
        """
        x_ = x @ self.P # (B, F)
        return torch.cat([torch.sin(x_), torch.cos(x_)], dim=1) # (B, 2F)
