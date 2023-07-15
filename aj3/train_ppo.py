import time

import numpy as np
import torch
from mltoolkit import parse_config
from tqdm import tqdm

from aj3.agents.ppo import PPOAgent
from aj3.configs import MazeArguments
from aj3.eval import evaluate_and_store_mp4
from aj3.net import MazeNet
from aj3.train import setup_seed, setup_gym, compute_stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup():
    cfg = MazeArguments(**parse_config('configs/maze.yaml'))
    setup_seed(cfg)
    env = setup_gym()
    return cfg, env


def main():
    cfg, env = setup()

    network = MazeNet(cfg, critic=True).to(device)
    print('Running training on device: {}'.format(device))
    agent = PPOAgent(env, network, cfg)

    # Training loop
    num_episodes = cfg.train.num_episodes
    results = []
    for episode in tqdm(range(num_episodes)):
        state = env.reset()

        states = []
        actions = []
        log_probs = []
        rewards = []

        start = time.time()
        for i in range(env._max_episode_steps):
            action_output = agent.act(state[None,])
            action, log_prob = action_output.action, action_output.log_prob
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            state = next_state

            if done:
                results.append((i + 1, i < env._max_episode_steps - 1))
                break
        end = time.time()
        print(f'Episode {episode + 1} took {end - start} seconds')

        start = time.time()
        # Compute advantages and returns
        with torch.no_grad():
            returns = []
            advantages = []
            G = 0
            for t in reversed(range(len(rewards))):
                G = rewards[t] + 0.99 * G  # Discount factor (gamma)
                returns.insert(0, G)
                s = torch.from_numpy(states[t]).long()
                advantages.insert(0, G - agent.net(s[None,])[1])

        # Normalize advantages
        advantages = torch.tensor(advantages, device=device)
        advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)

        # Update the policy
        agent.update_policy(states, actions, log_probs, returns, advantages)
        end = time.time()
        print(f'Update {episode + 1} took {end - start} seconds')

        if (episode + 1) % 100 == 0:
            avg_num_steps, avg_success_rate = compute_stats(results)
            print(f'Episode: {episode + 1}, Avg. num steps: {avg_num_steps}, Avg. success rate: {avg_success_rate}')
            results = []
            evaluate_and_store_mp4(env, agent, f'videos/{episode + 1}.mp4')

    evaluate_and_store_mp4(env, agent, f'videos/final.mp4')

    # Evaluate the agent
    state = env.reset()
    total_reward = 0
    done = False
    while not done:
        output = agent.act(state[None,])
        state, reward, done, _ = env.step(output.action)
        total_reward += reward

    print(f"Total reward: {total_reward}")


if __name__ == '__main__':
    main()
