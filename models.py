import torch 
from torch import nn 

import numpy as np


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


class Siren(nn.Module):
    def __init__(self, in_features=2, out_features=3, hidden_features = 256, hidden_layers =4, outermost_linear=False, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, x):
        return self.net(x)       

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))
    