
import numpy as np

from environments.qm_env_wrapper import GymEnvironment
from SAC.sac import SAC


def get_control_sequence(env : GymEnvironment, agent : SAC, seed = 0):

    obs, _ = env.reset(seed=seed)

    times = [float(env.qm_env.t)]

    actions = []

    final_info = {}

    while True:

        action = agent.act(obs, deterministic=True)

        obs, _, terminated, _, info = env.step(action)

        actions.append(float(np.asarray(action).reshape(-1)[0]))

        times.append(float(env.qm_env.t))

        final_info = info

        if terminated:
            break

    return (
        np.asarray(times, dtype=np.float64),
        np.asarray(actions, dtype=np.float32),
        final_info,
    )
