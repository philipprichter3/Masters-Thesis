"""Minimal SAC learner using the uploaded ActorCritic networks."""

from copy import deepcopy
from itertools import chain

import numpy as np
import torch

from SAC.core.actor_critic import ActorCritic


class SAC:

    def __init__(self, observation_space, action_space, *, hidden_sizes=(256, 256),
                 gamma=0.99, polyak=0.995, lr=3e-4, device='cpu'):
        
        self.device = torch.device(device)

        self.gamma = gamma
        self.polyak = polyak

        self.actor_critic = ActorCritic(
            observation_space, 
            action_space,
            hidden_sizes=hidden_sizes
        ).to(self.device)

        self.q1_target = deepcopy(self.actor_critic.q1).to(self.device)
        self.q2_target = deepcopy(self.actor_critic.q2).to(self.device)

        for p in chain(self.q1_target.parameters(), self.q2_target.parameters()):
            p.requires_grad_(False)

        self.q_params = list(
            chain(
                self.actor_critic.q1.parameters(), 
                self.actor_critic.q2.parameters()
                )
            )
        
        self.q_optimizer = torch.optim.Adam(self.q_params, lr=lr)

        self.pi_optimizer = torch.optim.Adam(self.actor_critic.pi.parameters(), lr=lr)

        self.alpha_optimizer = torch.optim.Adam(self.actor_critic.alpha.parameters(), lr=lr)

        self.target_entropy = -float(np.prod(action_space.shape))

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

    def update(self, batch):

        batch = {k: v.to(self.device) for k, v in batch.items()}

        obs, act, rew = batch['obs'], batch['act'], batch['rew']
        obs2, done = batch['next_obs'], batch['terminated']

        # Bellman target: use target critics, not the live critics.
        with torch.no_grad():

            next_act, next_logp = self.actor_critic.pi(obs2)

            next_q = torch.minimum(self.q1_target(obs2, next_act),
                                   self.q2_target(obs2, next_act))
            
            target_q = rew + self.gamma * (1 - done) * (
                next_q - self.actor_critic.alpha() * next_logp)

        q1 = self.actor_critic.q1(obs, act)
        q2 = self.actor_critic.q2(obs, act)

        q_loss = ((q1 - target_q).square() + (q2 - target_q).square()).mean()

        self.q_optimizer.zero_grad(set_to_none=True)

        q_loss.backward()

        self.q_optimizer.step()

        # Freeze critic weights but still differentiate Q(s, pi(s)) wrt actions.
        for p in self.q_params:

            p.requires_grad_(False)

        pi_act, logp = self.actor_critic.pi(obs)

        q_pi = torch.minimum(
            self.actor_critic.q1(obs, pi_act), 
            self.actor_critic.q2(obs, pi_act)
        )

        pi_loss = (self.actor_critic.alpha().detach() * logp - q_pi).mean()

        self.pi_optimizer.zero_grad(set_to_none=True)

        pi_loss.backward()

        self.pi_optimizer.step()

        for p in self.q_params:

            p.requires_grad_(True)

        # Temperature learns towards the target entropy.
        alpha_loss = -(
            self.actor_critic.alpha.log_alpha_val 
            * (logp.detach() + self.target_entropy)
        ).mean()

        self.alpha_optimizer.zero_grad(set_to_none=True)

        alpha_loss.backward()

        self.alpha_optimizer.step()

        # Polyak/EMA target updates.
        with torch.no_grad():
            for live, target in [(self.actor_critic.q1, self.q1_target),
                                 (self.actor_critic.q2, self.q2_target)]:
                
                for p, p_target in zip(live.parameters(), target.parameters()):
                    p_target.mul_(self.polyak).add_(p, alpha=1 - self.polyak)

        return {
            'q_loss': float(q_loss.detach()),
            'pi_loss': float(pi_loss.detach()),
            'alpha': float(self.actor_critic.alpha().detach())
        }
