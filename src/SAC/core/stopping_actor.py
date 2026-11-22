import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from rl_utils.mlp import mlp
from rl_utils.rescale_inputs import RescaleInput
from sac.core.actor import Actor


class StoppingActor(nn.Module):
    def __init__(
            self, 
            observation_space, 
            action_space, 
            hidden_sizes, 
            activation,
            min_cov_eigen=1.e-8, 
            initial_stop_probability=0.05
        ):
        
        super().__init__()

        class _ControlSpace:

            shape = (1,)

            def __init__(self, source):
                self.low = np.asarray(source.low[:1], dtype=np.float32)
                self.high = np.asarray(source.high[:1], dtype=np.float32)

        self.control = Actor(
            observation_space, 
            _ControlSpace(action_space), 
            hidden_sizes,
            activation, 
            min_cov_eigen=min_cov_eigen
        )

        obs_dim = observation_space.shape[0]

        self.rescale_input = RescaleInput(
            observation_space.low, 
            observation_space.high
        )

        self.stop_net = mlp([obs_dim] + list(hidden_sizes), activation, activation)

        self.stop_logit = nn.Linear(hidden_sizes[-1], 1)

        nn.init.zeros_(self.stop_logit.weight)

        prior = np.log(
            initial_stop_probability 
            / (1.0 - initial_stop_probability)
        )

        nn.init.constant_(self.stop_logit.bias, float(prior))

    def components(self, obs, deterministic=False):

        control, logp_u = self.control(
            obs, 
            deterministic=deterministic,
            with_logprob=True
        )

        net_out = self.rescale_input(obs)
        net_out = self.stop_net(net_out)

        logits = self.stop_logit(net_out).squeeze(-1)

        p_stop = torch.sigmoid(logits)

        expected_logp_stop = (
            p_stop * F.logsigmoid(logits)
            + (1.0 - p_stop) * F.logsigmoid(-logits)
        )

        return control, logp_u, p_stop, expected_logp_stop

    def forward(self, obs, deterministic=False, with_logprob=True):

        u, logp_u, p_stop, _ = self.components(
            obs, 
            deterministic=deterministic
        )

        if deterministic:
            stop = (p_stop > 0.5).float()
        else:
            stop = torch.bernoulli(p_stop)

        if with_logprob:

            p_safe = p_stop.clamp(
                min=1e-6, 
                max=1.0 - 1e-6
            )

            logp_stop = torch.where(
                stop > 0.5, torch.log(p_safe), torch.log1p(-p_safe)
            )

            logp = logp_u + logp_stop

        else:

            logp = None

        return torch.cat((u, stop.unsqueeze(-1)), dim=-1), logp
