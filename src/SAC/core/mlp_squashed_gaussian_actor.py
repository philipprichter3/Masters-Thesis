import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions.multivariate_normal import MultivariateNormal

from src.SAC.core.rescale_inputs import RescaleInput
from src.SAC.core.utils import mlp


class SquashedGaussianMLPActor(nn.Module):
    """
    Class representing the policy, implemented with fully connected layers
    that take the state as input, and outputs the averages and the covariance matrix
    of a multivariate normal distribution. The action is then computed by sampling 
    from this distribution, and passing the output thourgh a "squashing function" 
    (a hyperbolic tangent, and then a rescaling to the desired range).

    Args:
        observation_space(gym.spaces): observation space of the RL environment. Must be 1D 
            array        
        action_space(gym.spaces): action space of the RL environment. Must be 1D 
            array
        hidden_sizes(tuple(int)): list of sizes of hidden layers
        activation: activation function for all layers except for output
        min_cov_eigen (float): used for stability when computing the covariance matrix of the multivariate
            normal distribution.
    """
    def __init__(self, observation_space, action_space, hidden_sizes, activation,
                min_cov_eigen=1.e-8):
        super().__init__()

        #load state and action dimensions and bounds
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        obs_lower_bounds = observation_space.low
        obs_upper_bounds = observation_space.high
        self.register_buffer("act_lower_bounds", torch.tensor(action_space.low, dtype=torch.float32))
        self.register_buffer("act_upper_bounds", torch.tensor(action_space.high, dtype=torch.float32))
        self.act_dim = act_dim

        #define the input layer that places data in a bound
        self.rescale_input = RescaleInput(obs_lower_bounds, obs_upper_bounds)

        #main network taking the state as input, and passing it through all hidden layers
        self.net = mlp([obs_dim] + list(hidden_sizes), activation, activation)
        
        #output layer producing the average of the multivariate gaussina
        self.mu_layer = nn.Linear(hidden_sizes[-1], act_dim)
        
        #output layer producing a matrix M such that the covariance matrix is
        #M * M^T + min_cov_eigen * Id, where min_cov_eigen is a small constant to ensure strict positivity
        self.m_layer = nn.Linear(hidden_sizes[-1], act_dim*act_dim)
        
        #I create the matrix l_id = min_cov_eigen * Id for the covariance matrix
        self.register_buffer("l_id", torch.eye(act_dim)*min_cov_eigen)

    def forward(self, obs, deterministic=False, with_logprob=True):
        """
        Args:
            obs(torch.Tensor): batch of observations
            deterministic(bool): if the actions should be chosen deterministally or not
            with_logprob(bool): if the log of the probability should be computed and returned

        Returns:
            pi_action(torch.Tensor): the chosen continuous actions
            logp_pi(torch.Tensor): the log probability of such continuous actions
        """
        #run the state through the network
        net_out = self.rescale_input(obs)
        net_out = self.net(net_out)
        mu = self.mu_layer(net_out)
        m = self.m_layer(net_out).view(-1,self.act_dim,self.act_dim)
        cov_mat = (torch.transpose(m,1,2) @ m) + self.l_id

        # Pre-squash (tanh) distribution and sample
        pi_distribution = MultivariateNormal(mu, cov_mat)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()
        
        #if necessary, compute the log of the probabilities
        if with_logprob:
            logp_pi = pi_distribution.log_prob(pi_action)

            #change of distribution when going from gaussian to Tanh
            logp_pi -= (2.*(np.log(2.) - pi_action - F.softplus(-2.*pi_action))).sum(axis=1)
            
            #notice that we don't account for the rescaling of the action in logp_pi. This is to
            #make the entropy more universal, and not dependent on the specific interval size.
        else:
            logp_pi = None
       
        #apply tanh to the sampled gaussian
        pi_action = torch.tanh(pi_action)

        #apply the shift of the action to the correct interval
        pi_action = self.act_lower_bounds + 0.5*(pi_action + 1.)*(self.act_upper_bounds-self.act_lower_bounds)

        return pi_action, logp_pi