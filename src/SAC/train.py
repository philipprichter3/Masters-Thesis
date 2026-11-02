"""Minimal environment interaction and SAC training loop."""

from SAC.environment.qm_env_wrapper import GymEnvironment
from SAC.sac import SAC
from SAC.utils.replay_buffer import ReplayBuffer


def train(
        env : GymEnvironment, 
        hyper_params,
        seed = 0,
        steps=5_000, 
    ):
    
    agent = SAC(
        env.observation_space, 
        env.action_space, 

        use_cuda=hyper_params["USE_CUDA"],
        gamma=hyper_params["GAMMA"],
        polyak=hyper_params["POLYAK"],
        hidden_sizes=hyper_params["HIDDEN_SIZES"],
        min_cov_eigen=hyper_params["MIN_COV_EIGEN"],
        lr = hyper_params["LR"],
        lr_alpha= hyper_params["ALPHA_LR"]
    )

    memory = ReplayBuffer(
        env.observation_space, 
        env.action_space, 

        capacity=hyper_params["REPLAY_MEMORY_SIZE"]
    )

    obs, _ = env.reset(seed=seed)

    start_steps = hyper_params["INITIAL_RANDOM_STEPS"]

    batch_size = hyper_params["BATCH_SIZE"]

    update_after = hyper_params["UPDATE_AFTER"]
    update_every = hyper_params["UPDATE_EVERY"]

    ep_return = 0.0
    ep_length = 0
    episodes = 0

    for t in range(steps):

        action = (env.action_space.sample() 
                  if t < start_steps
                  else agent.act(obs))
        
        next_obs, reward, terminated, truncated, _ = env.step(action)

        memory.add(
            obs, 
            action, 
            reward, 
            next_obs, 
            terminated
        )
        
        ep_return += reward
        ep_length += 1
        obs = next_obs

        if t >= update_after and \
            len(memory) >= batch_size and \
            t % update_every == 0:

            losses = agent.update(
                memory.sample(
                    batch_size, 
                    device=agent.device
                )
            )

        if terminated or truncated:

            episodes += 1

            print(f'episode={episodes} steps={ep_length} '
                  f'return={ep_return:.5f}' +
                  (f' alpha={losses["alpha"]:.4f}' 
                   if t >= update_after and len(memory) >= batch_size 
                   else '')
                )
            
            obs, _ = env.reset()
            ep_return = 0.0
            ep_length = 0

    return agent
