import torch
from torch import nn

from sac.core.actor import Actor
from sac.core.alpha import Alpha
from sac.core.qfunction import QFunction
from sac.core.stopping_actor import StoppingActor


class ActorCritic(nn.Module):

    def __init__(
            self, 
            observation_space, 
            action_space, 
            hidden_sizes=(256,256),
            activation=nn.ReLU, 
            min_cov_eigen = 1.e-8,
            stopping = False,
            initial_stop_probability=0.05
        ):

        super().__init__()

        actor_kwargs = {
            "observation_space": observation_space,
            "action_space": action_space,
            "hidden_sizes": hidden_sizes,
            "activation": activation,
            "min_cov_eigen": min_cov_eigen
        }

        if stopping:

            self.pi = StoppingActor(
                **actor_kwargs,
                initial_stop_probability=initial_stop_probability
            )
            
        else:

            self.pi = Actor(**actor_kwargs)

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