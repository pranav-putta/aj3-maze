import os

import gym.vector
import torch
from gym import Env
from gym.vector import AsyncVectorEnv, SyncVectorEnv
from hydra.utils import instantiate
from tqdm import tqdm

from mazelens.agents import Agent
from mazelens.envs.rollout_env_wrapper import RolloutEnvWrapper
from mazelens.util import compute_returns


class Trainer:
    env: RolloutEnvWrapper
    base_env: Env
    agent: Agent

    def __init__(self, device, seed, exp_dir, agent=None, env=None, epochs=None,
                 num_rollout_steps=None, eval_frequency=None,
                 num_environments=None, lr=None, max_grad_norm=None,
                 gamma=None, tau=None, use_gae=None, log_videos=None):
        self.device = device
        self.seed = seed
        self.exp_dir = exp_dir
        self.agent_f = agent
        self.env_f = env

        self.epochs = epochs
        self.num_rollout_steps = num_rollout_steps
        self.eval_frequency = eval_frequency
        self.num_envs = num_environments
        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.tau = tau
        self.use_gae = use_gae
        self.log_videos = log_videos

    def process_rollouts(self, rollouts):
        return rollouts.to_tensordict()

    def init_train(self):
        self.env = RolloutEnvWrapper(
            SyncVectorEnv([lambda: self.env_f(seed=self.seed) for _ in range(self.num_envs)]))
        self.base_env = self.env_f(seed=self.seed)
        self.agent = self.agent_f(action_space=self.base_env.action_space,
                                  observation_space=self.base_env.observation_space,
                                  device=self.device)

        os.makedirs(os.path.join(self.exp_dir, 'checkpoints'), exist_ok=True)
        os.makedirs(os.path.join(self.exp_dir, 'videos'), exist_ok=True)

    def train(self):
        self.init_train()
        self.agent.to(self.device)

        print(
            f'Starting train with {type(self.agent).__name__} in'
            f' {type(self.base_env).__name__} on {self.device} device')

        for epoch in tqdm(range(self.epochs)):
            with torch.no_grad():
                rollouts = self.env.rollout(agent=self.agent, num_steps=self.num_rollout_steps)

            self.agent.learn(self.process_rollouts(rollouts))

            if (epoch + 1) % self.eval_frequency == 0:
                stats = rollouts.compute_stats(self.gamma)
                print(f'Stats for epoch {epoch + 1}: {stats}')

                if self.log_videos:
                    rollouts.save_episode_to_mp4(os.path.join(self.exp_dir, 'videos', f'epoch_{epoch + 1}.mp4'))
