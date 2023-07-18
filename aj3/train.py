import contextlib
import os
from dataclasses import dataclass
from pprint import pprint

import gym
import torch
from gym.wrappers import TimeLimit
from mltoolkit import parse_args
from tensordict import TensorDict
from tqdm import tqdm

from aj3.util.configs import MazeArguments
from aj3.maze.maze import Env
from aj3.util.util import get_trainer, get_policy_net, setup_seed


def setup_gym(cfg: MazeArguments):
    env_id = "Maze-v1"
    gym.register(id=env_id, entry_point=Env, max_episode_steps=cfg.env.max_steps, kwargs={'cfg': cfg})
    return env_id


def setup(cfg):
    setup_seed(cfg.seed)
    env_id = setup_gym(cfg)
    env = gym.make(env_id)

    network = get_policy_net(cfg.net_name)
    trainer = get_trainer(cfg.trainer_name)
    agent = trainer(env, network(cfg), cfg)
    agent = agent.to(cfg.device)

    print('Running training on device: {}'.format(cfg.device))
    print(f'Number of agent parameters {sum(p.numel() for p in agent.parameters())}')

    os.makedirs(cfg.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.exp_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(cfg.exp_dir, 'checkpoints'), exist_ok=True)

    envs = gym.vector.AsyncVectorEnv(
        [lambda: TimeLimit(Env(cfg), max_episode_steps=cfg.env.max_steps) for _ in range(cfg.train.num_envs)])
    return agent, cfg, envs


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


def train(cfg):
    agent, cfg, envs = setup(cfg)

    # Training loop
    pbar = tqdm(enumerate(range(cfg.train.num_steps)), total=cfg.train.num_steps)

    for i, episode in pbar:
        with rollout_mode(cfg):
            rollout = agent.collect_episodes(envs)
        agent.update_policy(rollout)
        stats = rollout.compute_stats(rollout.to_batch())

        if (i + 1) % cfg.log_every == 0:
            pprint(stats)
            rollout.save_episode_to_mp4(os.path.join(cfg.exp_dir, 'videos', f'episode_{i + 1}.mp4'))
            agent.save(os.path.join(cfg.exp_dir, 'checkpoints', f'ckpt.{i + 1}.pt'))


def main():
    cfg: MazeArguments = parse_args(MazeArguments)
    train(cfg)


if __name__ == '__main__':
    main()
