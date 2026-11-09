
import random

import numpy as np
import torch
from tqdm.auto import trange

from debug.training_logger import TrainingLogger
from environments.qm_env_wrapper import GymEnvironment
from sac.sac import SAC
from serialization.serialize_experiment import save_experiment
from utils.replay_buffer import ReplayBuffer
from utils.schedules import objective_weights, target_entropy

# ======================================================================
# ============================= INIT TRAINING ============================
# ======================================================================

def init(
        seed,
        env : GymEnvironment,
        hyper_params : dict
    ):

    #create agent
    agent = SAC(
        env.observation_space, 
        env.action_space, 

        use_cuda=hyper_params["USE_CUDA"],

        gamma=hyper_params["GAMMA"],
        polyak=hyper_params["POLYAK"],

        hidden_sizes=hyper_params["HIDDEN_SIZES"],

        min_cov_eigen=hyper_params["MIN_COV_EIGEN"],

        lr = hyper_params["LR"],
        lr_alpha = hyper_params["ALPHA_LR"],
    )

    #create buffer
    memory = ReplayBuffer(
        env.observation_space, 
        env.action_space, 

        capacity=hyper_params["REPLAY_MEMORY_SIZE"],

        seed = seed
    )

    return agent, memory

# ======================================================================
# ============================= TRAINING ============================
# ======================================================================

def train(
        env : GymEnvironment, 
        hyper_params : dict,
        seed = 0,        
        steps=5_000,
        *,        
        env_params: dict | None = None,
        serialize: bool = False,
    ) -> SAC:

    #init all rng

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    env.action_space.seed(seed)

    #create agent and replay buffer
    
    agent, memory = init(
        seed, 
        env, 
        hyper_params
    )

    #init all necessary params

    start_steps = hyper_params["INITIAL_RANDOM_STEPS"]

    batch_size = hyper_params["BATCH_SIZE"]

    update_after = hyper_params["UPDATE_AFTER"]
    update_every = hyper_params["UPDATE_EVERY"]

    c_start=hyper_params["C_START"]
    c_end=hyper_params["C_END"]
    c_width=hyper_params["C_WIDTH"]
    c_mean= hyper_params["C_MEAN"]

    h_start=hyper_params["H_START"]
    h_end=hyper_params["H_END"]
    h_decay=hyper_params["H_DECAY"]

    obs, _ = env.reset(seed=seed)

    #logging

    logger = TrainingLogger(hyper_params)

    #training loop

    for t in trange(steps, desc="SAC Training", unit="step"):

        action = (
            env.action_space.sample()
            if t < start_steps
            else agent.act(obs)
        )

        next_obs, _, terminated, _, info = env.step(action)

        logger.record_step(info)

        memory.add(
            obs, action, info["objectives"], next_obs, terminated
        )


        obs = next_obs

        global_step = t + 1

        if (
            global_step > update_after
            and global_step % update_every == 0
            and len(memory) >= batch_size
        ):
            
            weights = objective_weights(
                global_step, 
                c_start,
                c_end,
                c_width,
                c_mean
            )

            entropy = target_entropy(
                global_step, 
                h_start,
                h_end,
                h_decay
            )

            for _ in range(update_every):

                metrics = agent.update(
                    memory.sample(batch_size, device=agent.device),
                    reward_weights=weights,
                    target_entropy=entropy,
                )

                logger.record_update(metrics)

        if terminated:

            obs, _ = env.reset()

            logger.end_episode(global_step)

        logger.update(global_step)

    #serialization

    if serialize:

        directory = save_experiment(
            agent,
            env_params=env_params,
            hyper_params=hyper_params,
            seed=seed,
            steps=steps,
        )

    return agent

