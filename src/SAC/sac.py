
from copy import deepcopy
from itertools import chain
from pathlib import Path

import numpy as np
import torch

from sac.core.actor_critic import ActorCritic


class SAC:

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(
            self, 
            observation_space, 
            action_space,

            use_cuda = False,

            gamma = 0.993,
            polyak = 0.995,

            hidden_sizes = (512, 256),

            min_cov_eigen = 1.e-7,
            
            lr = 0.001,
            lr_alpha = 0.003,
        ):

        #initialize params
        
        self.device = torch.device(
            "cuda" if use_cuda
            else "cpu"
        )

        self.gamma = gamma
        self.polyak = polyak

        self.hidden_sizes = tuple(hidden_sizes)
        self.min_cov_eigen = float(min_cov_eigen)
        
        #initialize NNs

        self.actor_critic = ActorCritic(
            observation_space, 
            action_space,
            hidden_sizes=hidden_sizes,
            min_cov_eigen=min_cov_eigen
        ).to(self.device)

        self.q1_target = deepcopy(self.actor_critic.q1).to(self.device)
        self.q2_target = deepcopy(self.actor_critic.q2).to(self.device)

        for p in chain(self.q1_target.parameters(), self.q2_target.parameters()):
            p.requires_grad_(False)

        self.q_params = list(chain(
            self.actor_critic.q1.parameters(),
            self.actor_critic.q2.parameters(),
        ))

        #initialize optimizers

        self.q_optimizer = torch.optim.Adam(self.q_params, lr=lr)

        self.pi_optimizer = torch.optim.Adam(self.actor_critic.pi.parameters(), lr=lr)

        self.alpha_optimizer = torch.optim.SGD(self.actor_critic.alpha.parameters(), lr=lr_alpha)

# ======================================================================
# ============================= ACTION ============================
# ======================================================================

    def act(self, observation, deterministic=False):

        obs = torch.as_tensor(
            observation, 
            dtype=torch.float32, 
            device=self.device
        )

        return self.actor_critic.act(
            obs, 
            deterministic=deterministic
        ).cpu().numpy().astype(np.float32)

# ======================================================================
# ============================= UPDATE ============================
# ======================================================================

    def update(self, batch, *, reward_weights, target_entropy):

        batch = {key: value.to(self.device) for key, value in batch.items()}

        #retrieve data from batch
        obs, act = batch["obs"], batch["act"]
        next_obs, terminated = batch["next_obs"], batch["terminated"]
        objectives = batch["objectives"]  

        #apply weighting to objectives (blending function to energy vs ergotropy)
        weights = torch.as_tensor(
            reward_weights, device=self.device, dtype=objectives.dtype
        )

        reward = objectives @ weights

        with torch.no_grad():

            next_act, next_logp = self.actor_critic.pi(next_obs)

            next_q = torch.minimum(
                self.q1_target(next_obs, next_act),
                self.q2_target(next_obs, next_act),
            )

            #bellman function
            target_q = reward + self.gamma * (1.0 - terminated) * (
                next_q - self.actor_critic.alpha() * next_logp
            )

        #optimize critic functions
        q1 = self.actor_critic.q1(obs, act)
        q2 = self.actor_critic.q2(obs, act)

        q_loss = ((q1 - target_q).square() + (q2 - target_q).square()).mean()

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        for p in self.q_params:
            p.requires_grad_(False)

        #optimize actor (policy) 
        pi_act, logp = self.actor_critic.pi(obs)

        q_pi = torch.minimum(
            self.actor_critic.q1(obs, pi_act),
            self.actor_critic.q2(obs, pi_act),
        )

        pi_loss = (self.actor_critic.alpha().detach() * logp - q_pi).mean()

        self.pi_optimizer.zero_grad(set_to_none=True)
        pi_loss.backward()
        self.pi_optimizer.step()

        for p in self.q_params:
            p.requires_grad_(True)

        #optimize alpha (temperature)
        with torch.no_grad():
            _, logp_for_alpha = self.actor_critic.pi(obs)

        alpha_loss = -self.actor_critic.alpha() * (
            logp_for_alpha + float(target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_optimizer.step()

        #polyak average critic and target critic functions
        with torch.no_grad():
            for live, target in (
                (self.actor_critic.q1, self.q1_target),
                (self.actor_critic.q2, self.q2_target),
            ):
                for p, p_target in zip(live.parameters(), target.parameters()):
                    p_target.mul_(self.polyak).add_(
                        p, alpha=1.0 - self.polyak
                    )

        return {
            # Optimization
            "q_loss": q_loss.detach().item(),
            "pi_loss": pi_loss.detach().item(),
            "alpha_loss": alpha_loss.detach().item(),
            "alpha": self.actor_critic.alpha().detach().item(),

            # Critic diagnostics
            "q1": q1.detach().mean().item(),
            "q2": q2.detach().mean().item(),
            "q_diff": (q1.detach() - q2.detach()).abs().mean().item(),
            "target_q": target_q.detach().mean().item(),
            "td_error": (
                0.5 * (
                    (q1.detach() - target_q).abs() +
                    (q2.detach() - target_q).abs()
                )
            ).mean().item(),

            # Rewards
            "reward": reward.detach().mean().item(),

            # Policy diagnostics
            "q_pi": q_pi.detach().mean().item(),
            "entropy": -logp.detach().mean().item(),
            "target_entropy": float(target_entropy),
        }

# ======================================================================
# ============================= SERIALIZATION ============================
# ======================================================================

    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            {
                "hidden_sizes": self.hidden_sizes,
                "min_cov_eigen": self.min_cov_eigen,
                "policy": {
                    name: tensor.detach().cpu()
                    for name, tensor in self.actor_critic.pi.state_dict().items()
                },
            },
            path,
        )

    @classmethod
    def load(
        cls, 
        path, 
        observation_space, 
        action_space, 
        *, 
        use_cuda=False
    ):

        checkpoint = torch.load(
            path, 
            map_location="cpu",
            weights_only=True
        )

        agent = cls(
            observation_space,
            action_space,
            use_cuda=use_cuda,
            hidden_sizes=tuple(checkpoint["hidden_sizes"]),
            min_cov_eigen=float(checkpoint["min_cov_eigen"]),
        )

        agent.actor_critic.pi.load_state_dict(checkpoint["policy"])
        agent.actor_critic.pi.eval()
        
        return agent