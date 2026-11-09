
import gymnasium as gym
import numpy as np

from environments.qm_env import Environment


class GymEnvironment(gym.Env):

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(self, qm_env : Environment, params : dict):

        self.min_u = params["min_u"]
        self.max_u = params["max_u"]

        self.objective = params["objective"]
        self.reward_coeff = params["reward_coeff"]

        self.qm_env = qm_env

        self.last_u = self.min_u
        
        self.action_space = gym.spaces.Box(
            low=self.min_u,
            high=self.max_u,
            shape=(1,),
            dtype=np.float32
        )

        init_observation = self.observe()

        n = len(init_observation) - 2

        obs_low = np.concatenate([
            np.full(n, -1.0),
            [self.min_u, 0.0]
        ]).astype(np.float32)

        obs_high = np.concatenate([
            np.full(n, 1.0),
            [self.max_u, qm_env.tau]
        ]).astype(np.float32)

        self.observation_space = gym.spaces.Box(
            low = obs_low, 
            high = obs_high, 
            dtype = np.float32
        )

    def reset(self, *, seed = None, options = None):

        super().reset(seed=seed)

        self.qm_env.reset()
        self.last_u = self.min_u

        return self.observe(), {}

# ======================================================================
# ============================= RUNTIME ============================
# ======================================================================

    def step(self, action):

        action_array = np.asarray(
            action, dtype=np.float32
        ).reshape(-1)

        if not self.action_space.contains(action_array):
            raise ValueError(f"Action {action} out of bounds")

        u = float(action_array[0])

        info = dict(self.qm_env.step(u))
        info["objectives"] = self.reward(info)

        self.last_u = u
        observation = self.observe()

        terminated = bool(
            self.qm_env.t >= self.qm_env.tau - 1e-12
        )

        return observation, 0.0, terminated, False, info

# ======================================================================
# ============================= UTILS ============================
# ======================================================================

    def observe(self) -> np.ndarray:

        observation = self.qm_env.observe()

        return np.concatenate((
            observation,
            np.array([self.last_u, self.qm_env.t])
        )).astype(np.float32)

    def reward(self, info : dict):

        reward = self.qm_env.reward(info, self.objective)

        return self.reward_coeff * reward