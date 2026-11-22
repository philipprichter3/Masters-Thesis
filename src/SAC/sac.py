
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

        use_cuda=False,

        gamma=0.993,
        polyak=0.995,

        hidden_sizes=(512, 256),
        min_cov_eigen=1.e-7,

        lr=0.001,
        lr_alpha=0.003,

        stop_entropy_coef=0.0,
        initial_stop_probability=0.05,
    ):

        #initialize params

        self.device = torch.device(
            "cuda" if use_cuda else "cpu"
        )

        self.gamma = float(gamma)
        self.polyak = float(polyak)

        self.hidden_sizes = tuple(hidden_sizes)
        self.min_cov_eigen = float(min_cov_eigen)

        self.stop_entropy_coef = float(stop_entropy_coef)
        self.initial_stop_probability = float(
            initial_stop_probability
        )

        #initialize NNs

        self.actor_critic = ActorCritic(
            observation_space,
            action_space,
            hidden_sizes=hidden_sizes,
            min_cov_eigen=min_cov_eigen,
            initial_stop_probability=initial_stop_probability,
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
        self.alpha_optimizer = torch.optim.SGD(self.actor_critic.alpha.parameters(),lr=lr_alpha,)


# ======================================================================
# ============================= ACTION ============================
# ======================================================================

    def act(self, observation, deterministic=False):

        obs = torch.as_tensor(
            observation,
            dtype=torch.float32,
            device=self.device,
        )

        return self.actor_critic.act(
            obs,
            deterministic=deterministic,
        ).cpu().numpy().astype(np.float32)


# ======================================================================
# ============================= POLICY UTILS ============================
# ======================================================================

    def _policy_terms(self, obs, q1, q2):

        u, logp_u, p_stop, expected_logp_stop = (
            self.actor_critic.pi.components(obs)
        )

        stop_zero = torch.zeros_like(
            p_stop
        ).unsqueeze(-1)

        action_continue = torch.cat((u, stop_zero), dim=-1)
        action_stop = torch.cat((u, torch.ones_like(stop_zero)), dim=-1)

        q_continue = torch.minimum(
            q1(obs, action_continue).reshape(-1),
            q2(obs, action_continue).reshape(-1),
        )

        q_stop = torch.minimum(
            q1(obs, action_stop).reshape(-1),
            q2(obs, action_stop).reshape(-1),
        )

        expected_q = (
            (1.0 - p_stop) * q_continue
            + p_stop * q_stop
        )

        return (
            expected_q,
            logp_u,
            expected_logp_stop,
            p_stop,
        )

    def _logp_for_alpha(self, obs):
        """
        Temperature learning applies only to the
        continuous-action distribution.
        """

        _, logp_u, _, _ = (
            self.actor_critic.pi.components(obs)
        )
        return logp_u


# ======================================================================
# ============================= UPDATE ============================
# ======================================================================

    def update(self, batch, *, reward_weights, target_entropy):

        batch = {
            key: value.to(self.device)
            for key, value in batch.items()
        }

        #retrieve data from batch
        obs = batch["obs"]
        act = batch["act"]
        next_obs = batch["next_obs"]

        terminated = batch["terminated"].reshape(-1)
        objectives = batch["objectives"]

        #apply weighting to objectives (blending function to energy vs ergotropy)
        weights = torch.as_tensor(
            reward_weights,
            device=self.device,
            dtype=objectives.dtype,
        )

        reward = (objectives @ weights).reshape(-1)

        alpha = self.actor_critic.alpha()

        #bellman target
        with torch.no_grad():

            (next_q, next_logp_u, next_logp_stop, _,) = self._policy_terms(
                next_obs,
                self.q1_target,
                self.q2_target,
            )

            next_v = (
                next_q
                - alpha * next_logp_u
                - self.stop_entropy_coef * next_logp_stop
            )

            target_q = (
                reward
                + self.gamma
                * (1.0 - terminated)
                * next_v
            )

        #critic update

        q1 = self.actor_critic.q1(obs, act).reshape(-1)
        q2 = self.actor_critic.q2(obs, act).reshape(-1)

        q_loss = ((q1 - target_q).square() + (q2 - target_q).square()).mean()

        self.q_optimizer.zero_grad(set_to_none=True)
        q_loss.backward()
        self.q_optimizer.step()

        #actor update

        for p in self.q_params:
            p.requires_grad_(False)

        try:

            (q_pi, logp_u, expected_logp_stop, p_stop,) = self._policy_terms(
                obs,
                self.actor_critic.q1,
                self.actor_critic.q2,
            )

            pi_loss = (
                alpha.detach() * logp_u
                + self.stop_entropy_coef * expected_logp_stop
                - q_pi
            ).mean()

            self.pi_optimizer.zero_grad(set_to_none=True)
            pi_loss.backward()
            self.pi_optimizer.step()

        finally:

            for p in self.q_params:
                p.requires_grad_(True)

        #temperature update

        with torch.no_grad():
            logp_for_alpha = self._logp_for_alpha(obs)

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

                for p, p_target in zip(
                    live.parameters(),
                    target.parameters(),
                ):

                    p_target.mul_(self.polyak).add_(
                        p,
                        alpha=1.0 - self.polyak,
                    )

        diagnostics = {

            "q_loss": q_loss.detach().item(),
            "pi_loss": pi_loss.detach().item(),
            "alpha_loss": alpha_loss.detach().item(),
            "alpha": self.actor_critic.alpha().detach().item(),

            "q1": q1.detach().mean().item(),
            "q2": q2.detach().mean().item(),
            "q_diff": (
                q1.detach() - q2.detach()
            ).abs().mean().item(),
            "target_q": target_q.detach().mean().item(),
            "td_error": (
                0.5 * (
                    (q1.detach() - target_q).abs()
                    + (q2.detach() - target_q).abs()
                )
            ).mean().item(),

            "reward": reward.detach().mean().item(),

            "q_pi": q_pi.detach().mean().item(),
            "entropy": -logp_u.detach().mean().item(),
            "target_entropy": float(target_entropy),

            "stop_probability": p_stop.detach().mean().item(),
            "stop_entropy": -expected_logp_stop.detach().mean().item(),
            "stop_entropy_coef": self.stop_entropy_coef,
        }

        return diagnostics


# ======================================================================
# ============================= SERIALIZATION ============================
# ======================================================================

    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(
            self.actor_critic.pi.state_dict(),
            path,
        )

    @classmethod
    def load(
        cls,
        path,
        observation_space,
        action_space,
        *,
        hidden_sizes,
        use_cuda=False,
        initial_stop_probability=0.05,
    ):

        agent = cls(
            observation_space,
            action_space,
            use_cuda=use_cuda,
            hidden_sizes=hidden_sizes,
            initial_stop_probability=initial_stop_probability,
        )

        state_dict = torch.load(
            path,
            map_location="cpu",
            weights_only=True,
        )

        agent.actor_critic.pi.load_state_dict(state_dict)
        agent.actor_critic.pi.eval()

        return agent
