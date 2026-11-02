
import numpy as np
import torch


class ReplayBuffer:
    
    def __init__(
            self, 
            observation_space, 
            action_space, 
            capacity=100_000, 
            seed=None
        ):

        obs_shape = observation_space.shape
        act_shape = action_space.shape

        self.obs = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.empty((capacity, *obs_shape), dtype=np.float32)

        self.act = np.empty((capacity, *act_shape), dtype=np.float32)

        self.objectives = np.empty((capacity, 2), dtype=np.float32)

        self.terminated = np.empty(capacity, dtype=np.float32)

        self.capacity = capacity

        self.pos = 0

        self.size = 0

        self.rng = np.random.default_rng(seed)

    def add(
            self, 
            obs, 
            act, 
            objectives, 
            next_obs, 
            terminated
        ):

        i = self.pos

        self.obs[i] = obs
        self.next_obs[i] = next_obs

        self.act[i] = act

        self.objectives[i] = np.asarray(objectives, dtype=np.float32).reshape(-1)

        self.terminated[i] = terminated

        self.pos = (i + 1) % self.capacity

        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device="cpu"):

        idx = self.rng.integers(self.size, size=batch_size)

        return {
            "obs": torch.as_tensor(self.obs[idx], device=device),
            "next_obs": torch.as_tensor(self.next_obs[idx], device=device),
            "act": torch.as_tensor(self.act[idx], device=device),
            "objectives": torch.as_tensor(self.objectives[idx], device=device),
            "terminated": torch.as_tensor(self.terminated[idx], device=device),
        }

    def __len__(self):
        return self.size
