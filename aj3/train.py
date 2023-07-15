import torch
from mltoolkit import parse_config

from aj3.agents.rand import RandomAgent
from aj3.agents.reinforce import REINFORCEAgent
from aj3.configs import MazeArguments
from aj3.maze.maze import Env
import matplotlib.pyplot as plt
from tqdm import tqdm
import gym
import random
from eval import evaluate_and_store_mp4

from aj3.net import MazeNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def setup_seed(cfg):
    random.seed(cfg.seed)


def setup_gym():
    env_id = "Maze-v1"
    gym.envs.register(id=env_id, entry_point=Env, max_episode_steps=500)
    env = gym.make(env_id)
    return env


def train():
    pass


def compute_stats(results):
    num_steps = [r[0] for r in results if r[1]]
    num_successes = [r[1] for r in results]

    if len(num_steps) > 0:
        avg_num_steps = sum(num_steps) / len(num_steps)
    else:
        avg_num_steps = float('inf')
    avg_success_rate = sum(num_successes) / len(num_successes)

    return avg_num_steps, avg_success_rate


def main():
    cfg = MazeArguments(**parse_config('configs/maze.yaml'))
    setup_seed(cfg)
    env = setup_gym()

    net = MazeNet(cfg).to(device)
    agent = REINFORCEAgent(env, net)

    # Training loop
    results = []
    for episode in tqdm(range(cfg.train.num_episodes)):
        state = env.reset()
        rewards = []
        log_probs = []

        for i in range(env._max_episode_steps):  # Assuming env._max_episode_steps is the maximum number of steps
            state = state[None,]
            action, log_prob = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            rewards.append(reward)
            log_probs.append(log_prob)

            if done:
                results.append((i + 1, i < env._max_episode_steps - 1))
                break

            state = next_state

        agent.update_policy(rewards, log_probs)

        if (episode + 1) % 100 == 0:
            avg_num_steps, avg_success_rate = compute_stats(results)
            print(f'Episode: {episode + 1}, Avg. num steps: {avg_num_steps}, Avg. success rate: {avg_success_rate}')
            results = []
            evaluate_and_store_mp4(env, agent, f'videos/{episode + 1}.mp4')


if __name__ == '__main__':
    main()
