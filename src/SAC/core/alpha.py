import torch
from torch import nn


class Alpha(nn.Module):
    """
    This module is used to return the current value of the "temperature parameter" alpha, and to update
    it using gradient descent in order to ensure a given target entropy of the distribution. Since alpha>0,
    we store log_alpha, which can be any R number, and then exponentiate it to get alpha = e*log_alpha.
    This ensures that any update cannot make alpha negative.
       
    """
    def __init__(self):
        super().__init__()

        #initialize it to 1., an arbitrary initial value
        self.log_alpha_val = nn.Parameter(torch.tensor(1., dtype=torch.float32))
        
    def forward(self):
        """
        return the value of alpha
        """
        return torch.exp(self.log_alpha_val)