import torch
from torch import nn

from serialization.serializable_NN import Serializable__NN


class Alpha(Serializable__NN):

    def __init__(self):
        
        super().__init__()

        #initialize it to 1., an arbitrary initial value
        self.log_alpha_val = nn.Parameter(torch.tensor(1., dtype=torch.float32))
        
    def forward(self):
        """
        return the value of alpha
        """
        return torch.exp(self.log_alpha_val)