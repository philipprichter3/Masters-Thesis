import torch
from alpha import Alpha
from mlp_qfunction import MLPQFunction
from mlp_squashed_gaussian_actor import SquashedGaussianMLPActor
from torch import nn


class MLPActorCritic(nn.Module):
    """
    Module that contains two value functions, self.q1 and self.q2, the policy self.pi,
    and a module for the "temperature" parameter self.alpha.
    Both the value function and the policy are parameterizes using an arbitrary number of fully connected layers.
  
    Args:
        observation_space(gym.spaces): observation space of the environment (must be 1D array)
        action_space (gym.spaces): action space of sac_envs environments (must be 1D array)
        hidden_sizes (tuple(int)): size of each of the hidden layers that will be addded to 
            both the value and policy functions
        activation: activation function for all layers except for the output layer
        min_cov_eigen (float): used for stability when computing the covariance matrix of the multivariate
            normal distribution.

    """
    def __init__(self, observation_space, action_space, hidden_sizes=(256,256),
                 activation=nn.ReLU, min_cov_eigen = 1.e-8):
        super().__init__()

        # build policy, value functions, and module for the value of alpha
        self.pi = SquashedGaussianMLPActor(observation_space, action_space, hidden_sizes, activation,
                                            min_cov_eigen=min_cov_eigen)
        self.q1 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.q2 = MLPQFunction(observation_space, action_space, hidden_sizes, activation)
        self.alpha = Alpha()

    def act(self, obs, deterministic=False):
        """ return the action, chosen according to deterministic, given a single unbatched observation obs """
        with torch.no_grad():
            a, _ = self.pi(obs.view(1,-1), deterministic, False)
            return a[0]

    def alpha_no_grad(self):
        """
        returns the current value of the temperature parameter alpha without gradients
        """
        with torch.no_grad():
            return self.alpha()