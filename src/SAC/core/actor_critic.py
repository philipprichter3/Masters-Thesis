import torch
from torch import nn

from sac.core.alpha import Alpha
from sac.core.qfunction import QFunction
from sac.core.squashed_gaussian_actor import SquashedGaussianActor
from serialization.serializable_NN import Serializable__NN


class ActorCritic(Serializable__NN):

    def __init__(
            self, 
            observation_space, 
            action_space, 
            hidden_sizes=(256,256),
            activation=nn.ReLU, 
            min_cov_eigen = 1.e-8
        ):

        super().__init__()

        self.pi = SquashedGaussianActor(
            observation_space, 
            action_space, 
            hidden_sizes, 
            activation,
            min_cov_eigen=min_cov_eigen
        )

        self.q1 = QFunction(
            observation_space, 
            action_space, 
            hidden_sizes, 
            activation
        )

        self.q2 = QFunction(
            observation_space, 
            action_space, 
            hidden_sizes, 
            activation
        )

        self.alpha = Alpha()

    def act(self, obs, deterministic=False):
        """ 
        return the action, chosen according to deterministic, given a single unbatched observation obs 
        """

        with torch.no_grad():
            a, _ = self.pi(obs.view(1,-1), deterministic, False)
            return a[0]

    def alpha_no_grad(self):
        """
        returns the current value of the temperature parameter alpha without gradients
        """
        
        with torch.no_grad():
            return self.alpha()