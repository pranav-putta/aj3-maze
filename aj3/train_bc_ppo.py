import contextlib
import os
from pprint import pprint

import gym
import torch
from gym.wrappers import TimeLimit
from mltoolkit import parse_args
from tqdm import tqdm

from aj3.agents import PPOAgent
from aj3.maze.maze import Env
from aj3.maze.vec_env import VectorEnv
from aj3.util.configs import MazeArguments
from aj3.util.util import get_trainer, get_policy_net, setup_seed


def setup_gym(cfg: MazeArguments):
    env_id = "Maze-v1"
    gym.register(id=env_id, entry_point=Env, max_episode_steps=cfg.env.max_steps, kwargs={'cfg': cfg})
    return env_id


def setup(cfg):
    setup_seed(cfg.seed)
    env_id = setup_gym(cfg)
    env = gym.make(env_id)

    teacher_network = get_policy_net(cfg.train.bc_teacher_net)
    teacher_trainer = get_trainer(cfg.train.bc_teacher_trainer)
    teacher = teacher_trainer(env, teacher_network(cfg), cfg)

    bc_network = get_policy_net(cfg.net_name)
    bc_trainer = get_trainer(cfg.trainer_name)
    agent = bc_trainer(env, bc_network(cfg), cfg)

    agent = agent.to(cfg.device)
    teacher = teacher.to(cfg.device)
    print('Running training on device: {}'.format(cfg.device))
    print(f'Number of agent parameters {sum(p.numel() for p in agent.parameters())}')

    # create experiments dir to store things
    os.makedirs(cfg.exp_dir, exist_ok=True)
    os.makedirs(os.path.join(cfg.exp_dir, 'videos'), exist_ok=True)
    os.makedirs(os.path.join(cfg.exp_dir, 'checkpoints'), exist_ok=True)

    envs = VectorEnv(
        [lambda: TimeLimit(Env(cfg), max_episode_steps=cfg.env.max_steps) for _ in range(cfg.train.num_envs)])
    return agent, teacher, cfg, envs


def rollout_mode(cfg):
    if cfg.trainer_name == "ppo":
        return torch.no_grad()
    else:
        return contextlib.nullcontext()


def train(cfg):
    agent, teacher, cfg, envs = setup(cfg)

    # Training loop
    pbar = tqdm(enumerate(range(cfg.train.num_steps)), total=cfg.train.num_steps)

    for i, episode in pbar:
        with rollout_mode(cfg):
            if not cfg.train.single_rollout or i == 0:
                rollout = teacher.collect_episodes(envs)

        teacher.update_policy(rollout)
        agent.update_policy(rollout)

        if (i + 1) % cfg.log_every == 0:
            teacher_stats = rollout.compute_stats(rollout.to_batch())
            eval_rollout = agent.collect_episodes(envs)
            eval_stats = eval_rollout.compute_stats(eval_rollout.to_batch())
            eval_rollout.save_episode_to_mp4(os.path.join(cfg.exp_dir, 'videos', f'bc_episode_{i + 1}.mp4'))
            rollout.save_episode_to_mp4(os.path.join(cfg.exp_dir, 'videos', f'teacher_episode_{i + 1}.mp4'))

            pprint(f'Episode {i + 1} teacher stats: {teacher_stats}')
            pprint(f'Episode {i + 1} bc stats: {eval_stats}')

            agent.save(os.path.join(cfg.exp_dir, 'checkpoints', f'bc_agent.{i + 1}.ckpt.pt'))
            teacher.save(os.path.join(cfg.exp_dir, 'checkpoints', f'teacher_agent.{i + 1}.ckpt.pt'))


def main():
    cfg: MazeArguments = parse_args(MazeArguments)
    train(cfg)


if __name__ == '__main__':
    main()
