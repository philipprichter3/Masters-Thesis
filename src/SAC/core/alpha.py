import torch
from torch import nn


class Alpha(nn.Module):

    def __init__(self):
        
        super().__init__()

        #initialize it to 1., an arbitrary initial value
        self.log_alpha_val = nn.Parameter(torch.tensor(1., dtype=torch.float32))
        
    def forward(self):
        """
        return the value of alpha
        """
        return torch.exp(self.log_alpha_val)