import os
from dataclasses import asdict

import gym.vector
import torch
from gym import Env
from gym.vector import AsyncVectorEnv, SyncVectorEnv
from hydra.utils import instantiate
from tqdm import tqdm

from mazelens.agents import Agent
from mazelens.envs.rollout_env_wrapper import RolloutEnvWrapper
from mazelens.util import compute_returns
from mazelens.util.logger import Logger


class Trainer:
    env: RolloutEnvWrapper
    base_env: Env
    agent: Agent
    logger: Logger

    def __init__(self, device=None, seed=None, exp_dir=None, exp_name=None, logger=None, agent=None, env=None,
                 epochs=None,
                 num_rollout_steps=None, eval_frequency=None, num_environments=None, log_videos=None):
        self.device = device
        self.seed = seed
        self.exp_root_dir = exp_dir
        self.exp_name = exp_name
        self.agent_f = agent
        self.env_f = env

        self.epochs = epochs
        self.num_rollout_steps = num_rollout_steps
        self.eval_frequency = eval_frequency
        self.num_envs = num_environments
        self.log_videos = log_videos
        self.logger = logger

    def init_train(self):
        self.env = RolloutEnvWrapper(SyncVectorEnv([lambda: self.env_f(seed=self.seed) for _ in range(self.num_envs)]))
        self.base_env = self.env_f(seed=self.seed)
        self.agent = self.agent_f(action_space=self.base_env.action_space,
                                  observation_space=self.base_env.observation_space,
                                  device=self.device,
                                  epochs=self.epochs)

        os.makedirs(os.path.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'videos'), exist_ok=True)

        self.agent.to(self.device)

        print(
            f'Starting train with {type(self.agent).__name__} in'
            f' {type(self.base_env).__name__} on {self.device} device')
        if self.agent.parameters() is not None:
            print("Agent parameters: ", sum(p.numel() for p in self.agent.parameters() if p.requires_grad))
        else:
            print("Agent does not have parameters")

        print(f'Saving results to {self.exp_dir}')

    def train(self):
        self.init_train()
        stats = None

        for epoch in tqdm(range(self.epochs)):
            with torch.no_grad():
                rollouts = self.env.rollout(agent=self.agent, num_steps=self.num_rollout_steps)

            loss = self.agent.update(rollouts)
            stats = rollouts.compute_stats(0.99)

            self.logger.log({
                'epoch': epoch + 1,
                **asdict(stats), **loss
            })

            if (epoch + 1) % self.eval_frequency == 0:
                if self.log_videos:
                    rollouts.save_episode_to_mp4(os.path.join(self.exp_dir, 'videos', f'epoch_{epoch + 1}.mp4'))

        return stats

    @property
    def exp_dir(self):
        return os.path.join(self.exp_root_dir, self.exp_name)
