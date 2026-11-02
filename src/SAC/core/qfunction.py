import numpy as np
import torch
from torch import nn

from SAC.utils.mlp import mlp
from SAC.utils.rescale_inputs import RescaleInput


class QFunction(nn.Module):

    def __init__(
            self, 
            observation_space, 
            action_space, 
            hidden_sizes, 
            activation
        ):

        super().__init__()

        #determine action and observation spaces
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]

        obs_act_lower_bounds = np.concatenate([observation_space.low, action_space.low ])
        obs_act_upper_bounds = np.concatenate([observation_space.high, action_space.high ])

        #define the input layer that places data in a bound
        self.rescale_input = RescaleInput(
            obs_act_lower_bounds, 
            obs_act_upper_bounds
        )

        #define the main fully connected NN
        self.q = mlp(
            [obs_dim + act_dim] + list(hidden_sizes) + [1], 
            activation
        )

    def forward(self, obs, act):
        """
        Args:
            obs(torch.Tensor): batch of observations
            act(torch.Tensor): batch of continuous actions

        Returns:
            (torch.Tensor): 1D tensor with value of each state-action in the batch
        """
        
        out = self.rescale_input(torch.cat([obs, act], dim=-1))
        out = self.q(out)
        
        return torch.squeeze(out, -1) 