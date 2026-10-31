import torch
from torch import nn

#Constants used by the NNs to prevent numerical instabilities
UNIFORM_SIGMA =3.46


class RescaleInput(nn.Module):
    """
    This module rescales the input such that, if the input data was uniformly distributed
    between lower_bounds and upper_bouds, it would have variance 1. 
        
    Args:
        lower_bounds(np.array(float)): 1D array, with the same size of input, with all the
            lower bounds
        upper_bounds(np.array(float)): 1D array, with the same size of input, with all the
            upper bounds
    
    """
    def __init__(self, lower_bounds, upper_bounds):
        super().__init__()
        self.register_buffer("lower_bounds", torch.tensor(lower_bounds, dtype=torch.float32))
        self.register_buffer("upper_bounds", torch.tensor(upper_bounds, dtype=torch.float32))
    
    def forward(self, x):
        return ((x-self.lower_bounds)/(self.upper_bounds-self.lower_bounds)-0.5)*UNIFORM_SIGMA