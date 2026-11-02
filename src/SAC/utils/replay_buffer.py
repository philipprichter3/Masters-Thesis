import numpy as np
import torch


class ReplayBuffer:
    """Fixed-capacity, uniform off-policy transition buffer."""

    def __init__(self, observation_space, action_space, capacity=100_000):

        obs_shape = observation_space.shape
        act_shape = action_space.shape

        self.obs = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=np.float32)

        self.act = np.empty((capacity, *act_shape), dtype=np.float32)

        self.rew = np.empty(capacity, dtype=np.float32)

        self.terminated = np.empty(capacity, dtype=np.float32)

        self.capacity = capacity

        self.pos = 0

        self.size = 0

    def add(self, obs, act, rew, next_obs, terminated):

        i = self.pos

        self.obs[i] = obs
        self.next_obs[i] = next_obs

        self.act[i] = act

        self.rew[i] = rew

        self.terminated[i] = float(terminated)

        self.pos = (i + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device='cpu'):

        idx = np.random.randint(self.size, size=batch_size)

        return {
            'obs': torch.as_tensor(self.obs[idx], device=device),
            'next_obs': torch.as_tensor(self.next_obs[idx], device=device),

            'act': torch.as_tensor(self.act[idx], device=device),

            'rew': torch.as_tensor(self.rew[idx], device=device),

            'terminated': torch.as_tensor(self.terminated[idx], device=device),
        }

    def __len__(self):
        return self.size
