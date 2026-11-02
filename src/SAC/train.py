"""Minimal environment interaction and SAC training loop."""

from SAC.sac import SAC
from SAC.utils.replay_buffer import ReplayBuffer


def train(env, *, steps=5_000, start_steps=250, update_after=250,
          batch_size=64, replay_size=5_000, gamma=0.99):
    
    agent = SAC(
        env.observation_space, 
        env.action_space, 
        gamma=gamma
    )

    memory = ReplayBuffer(
        env.observation_space, 
        env.action_space, 
        capacity=replay_size
    )

    obs, _ = env.reset(seed=0)

    ep_return = 0.0
    ep_length = 0
    episodes = 0

    for t in range(steps):
        action = (env.action_space.sample() 
                  if t < start_steps
                  else agent.act(obs))
        
        obs2, reward, terminated, truncated, _ = env.step(action)

        memory.add(obs, action, reward, obs2, terminated)
        
        ep_return += reward
        ep_length += 1
        obs = obs2

        if t >= update_after and len(memory) >= batch_size:
            losses = agent.update(memory.sample(batch_size, device=agent.device))

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
