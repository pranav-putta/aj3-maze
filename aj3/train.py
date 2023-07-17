import cProfile
import contextlib
import time
from dataclasses import dataclass
from functools import wraps
from pprint import pprint

import gym
import numpy as np
import torch
from tensordict import TensorDict

from aj3.agents.reinforce import REINFORCEAgent
from mltoolkit import parse_config, parse_args
from tqdm import tqdm

from aj3.agents.ppo import PPOAgent
from aj3.configs import MazeArguments
from aj3.eval import evaluate_and_store_mp4
from aj3.maze.maze import Env
from aj3.nets.simple import SimplePolicyNet
from aj3.storage import RolloutStorage
from aj3.util import get_trainer, get_policy_net, setup_seed


def setup_gym(cfg: MazeArguments):
    env_id = "Maze-v1"
    gym.envs.register(id=env_id, entry_point=Env, max_episode_steps=cfg.env.max_steps, kwargs={'cfg': cfg})
    return env_id


def setup(cfg):
    setup_seed(cfg.seed)
    env_id = setup_gym(cfg)
    env = gym.make(env_id)

    network = get_policy_net(cfg)
    trainer = get_trainer(cfg)
    agent = trainer(env, network(cfg), cfg)
    agent = agent.to(cfg.device)

    print('Running training on device: {}'.format(cfg.device))
    return agent, cfg, env_id


@dataclass
class CollectionResult:
    batch: TensorDict
    total_episodes: int
    results: list


def rollout_mode(cfg):
    if cfg.trainer_name == "ppo":
        return torch.no_grad()
    else:
        return contextlib.nullcontext()


def collect_episodes_multiple_envs(envs, agent, cfg: MazeArguments):
    num_steps = cfg.train.max_steps
    rollouts = RolloutStorage(cfg)

    curr_state = [env.reset() for env in envs]
    hidden_state = torch.zeros((cfg.train.gru_layers, cfg.train.num_envs, cfg.train.hidden_size), dtype=torch.float,
                               device=cfg.device)
    for i in range(num_steps):
        curr_state = torch.tensor(np.array(curr_state))
        policy_out = agent.act(curr_state, hx=hidden_state)
        next_hidden_state, acts, log_prob = policy_out.hidden_state, policy_out.action, policy_out.log_prob
        next_states, rwd, dones, infos = zip(*[env.step(action) for env, action in zip(envs, acts)])
        success = [info['success'] for info in infos]

        rollouts.insert(curr_state, acts, log_prob, rwd, dones, hidden_state, success)
        next_states = [state if not done else env.reset() for state, done, env in zip(next_states, dones, envs)]

        curr_state = next_states
        hidden_state = next_hidden_state
        hidden_state *= ~torch.tensor(dones, device=cfg.device).repeat(cfg.train.gru_layers, cfg.train.hidden_size,
                                                                       1).transpose(1, 2)

    # push the last state in the rollouts
    rollouts.states.append(torch.tensor(np.array(curr_state), dtype=torch.long))
    return rollouts


def train(cfg):
    agent, cfg, env_id = setup(cfg)
    envs = [gym.make(env_id) for _ in range(cfg.train.num_envs)]

    # Training loop
    pbar = tqdm(enumerate(range(cfg.train.num_steps)), total=cfg.train.num_steps)

    for i, episode in pbar:
        with rollout_mode(cfg):
            rollout = collect_episodes_multiple_envs(envs, agent, cfg)
        agent.update_policy(rollout)
        stats = rollout.compute_stats(rollout.to_batch())

        if (i + 1) % cfg.log_every == 0:
            pprint(stats)
            evaluate_and_store_mp4(envs[0], agent, f'videos/{episode + 1}.mp4')


def main():
    cfg: MazeArguments = parse_args(MazeArguments)
    train(cfg)


if __name__ == '__main__':
    main()
