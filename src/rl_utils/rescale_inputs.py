import torch
from torch import nn

UNIFORM_SIGMA =3.46

class RescaleInput(nn.Module):

    def __init__(self, lower_bounds, upper_bounds):

        super().__init__()

        self.register_buffer(
            "lower_bounds", 
            torch.tensor(lower_bounds, dtype=torch.float32)
        )

        self.register_buffer(
            "upper_bounds", 
            torch.tensor(upper_bounds, dtype=torch.float32)
        )
    
    def forward(self, x):

        return (
            (x - self.lower_bounds)
            / (self.upper_bounds - self.lower_bounds)
            - 0.5
        ) * UNIFORM_SIGMA