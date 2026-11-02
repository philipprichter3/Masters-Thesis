
import gymnasium as gym
import numpy as np

from environments.qm_env import Environment


class GymEnvironment(gym.Env):

# ======================================================================
# ============================= INIT ============================
# ======================================================================

    def __init__(self, qm_env : Environment, params):

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

    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)

        self.qm_env.reset()

        self.last_u = self.min_u

        return \
            self.observe(), \
            {}
    
# ======================================================================
# ============================= OBSERVATIONS ============================
# ======================================================================

    #this will be abs method called from qm_env in future
    def observe(self) -> np.ndarray:

        rho = self.qm_env.rho.full().ravel()

        return np.concatenate((
            rho.real,
            rho.imag,
            np.array([self.last_u, self.qm_env.t]),
        )).astype(np.float32)

# ======================================================================
# ============================= RUNTIME ============================
# ======================================================================

    def step(self, action):

        action_array = np.asarray(action, dtype=np.float32).reshape(-1)

        if not self.action_space.contains(action):
            raise Exception(f"Action {action} out of bound")  # noqa: TRY002
        
        u = float(action_array[0])

        E_i = self.qm_env.energy()
        erg_i = self.qm_env.ergotropy(
            space=self.objective.removeprefix("ergotropy_")
        )

        info = self.qm_env.step(u)

        dE = float(info["energy"] - E_i)
        dErg = float(info[self.objective] - erg_i)

        multi_obj = self.reward_coeff * np.array(
            [dE, dErg], dtype=np.float32
        )
        self.last_u = u

        info = dict(info) 
        info["multi_obj"] = multi_obj

        terminated = bool(self.qm_env.t >= self.qm_env.tau - 1e-12)

        return \
            self.observe(), \
            terminated, \
            info