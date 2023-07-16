import time

import gym
import numpy as np
import torch
from mltoolkit import parse_config
from tqdm import tqdm

from aj3.agents.ppo import PPOAgent
from aj3.configs import MazeArguments
from aj3.eval import evaluate_and_store_mp4
from aj3.maze.maze import Env
from aj3.net import MazeNet
from aj3.train import setup_seed, compute_stats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_gym(cfg: MazeArguments):
    env_id = "Maze-v1"
    gym.envs.register(id=env_id, entry_point=Env, max_episode_steps=cfg.max_env_steps)
    return env_id


def setup():
    cfg = MazeArguments(**parse_config('configs/maze.yaml'))
    setup_seed(cfg)
    env_id = setup_gym(cfg)
    return cfg, env_id


@torch.no_grad()
def collect_episodes_multiple_envs(envs, agent, cfg: MazeArguments):
    states = []
    actions = []
    log_probs = []
    rewards = []
    results = []
    last_start_idx = [0 for _ in range(cfg.train.num_envs)]

    num_steps = cfg.train.max_steps
    max_episode_steps = envs[0]._max_episode_steps

    curr_state = [env.reset() for env in envs]

    for i in range(num_steps):
        policy_out = agent.act(np.array(curr_state)[:, None])
        acts, log_prob = policy_out.action, policy_out.log_prob
        next_states, rwd, dones, _ = zip(*[env.step(action) for env, action in zip(envs, acts)])

        states.append(curr_state)
        actions.append(acts)
        log_probs.append(log_prob)
        rewards.append(list(rwd))
        next_states = list(next_states)

        for j, done in enumerate(dones):
            if done:
                results.append((i + 1 - last_start_idx[j], i < max_episode_steps - 1))
                last_start_idx[j] = i + 1
                next_states[j] = envs[j].reset()

        curr_state = next_states

    # restructure data to be of shape (num_envs, num_steps, ...)
    states = torch.tensor(np.array(states)).transpose(1, 0)
    actions = torch.stack(actions).transpose(1, 0)
    log_probs = torch.stack(log_probs).transpose(1, 0)
    rewards = torch.tensor(rewards).transpose(1, 0)

    # compute advantages and returns
    returns = []
    advantages = []

    G = 0
    for t in reversed(range(num_steps)):
        G = rewards[:, t:t + 1] + 0.99 * G  # Discount factor (gamma)
        returns.append(G)
        advantages.append(G - agent.net(states[:, t:t + 1])[1].cpu())

    returns = torch.flip(torch.stack(returns).transpose(1, 0), dims=(1,)).squeeze(-1)
    advantages = torch.flip(torch.stack(advantages).transpose(1, 0), dims=(1,)).squeeze(-1)
    advantages = (advantages - torch.mean(advantages)) / (torch.std(advantages) + 1e-8)

    return states, actions, log_probs, returns, advantages, results


def main():
    cfg, env_id = setup()
    envs = [gym.make(env_id) for _ in range(cfg.train.num_envs)]

    network = MazeNet(cfg, critic=True).to(device)
    print('Running training on device: {}'.format(device))
    agent = PPOAgent(envs[0], network, epsilon=cfg.train.epsilon, entropy_coef=cfg.train.entropy_coef,
                     value_coef=cfg.train.value_coef)

    # Training loop
    num_episodes = cfg.train.num_episodes
    results = []
    for episode in tqdm(range(num_episodes)):

        states, actions, log_probs, returns, advantages, result = collect_episodes_multiple_envs(envs, agent, cfg)
        results.extend(result)

        # Update the policy
        agent.update_policy(states, actions, log_probs, returns, advantages)

        if (episode + 1) % cfg.log_every == 0:
            avg_num_steps, avg_success_rate = compute_stats(results)
            print(f'Episode: {episode + 1}, Avg. num steps: {avg_num_steps}, Avg. success rate: {avg_success_rate}')
            results = []
            evaluate_and_store_mp4(envs[0], agent, f'videos/{episode + 1}.mp4')

    evaluate_and_store_mp4(envs[0], agent, f'videos/final.mp4')

    # Evaluate the agent
    state = envs[0].reset()
    total_reward = 0
    done = False
    while not done:
        output = agent.act(state[None, None, ...])
        state, reward, done, _ = envs[0].step(output.action)
        total_reward += reward

    print(f"Total reward: {total_reward}")


if __name__ == '__main__':
    main()
