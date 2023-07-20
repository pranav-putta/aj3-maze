import os

import gym.vector
import torch
from gym import Env
from gym.vector import AsyncVectorEnv, SyncVectorEnv
from hydra.utils import instantiate
from tqdm import tqdm

from mazelens.agents import Agent
from mazelens.envs.rollout_env_wrapper import RolloutEnvWrapper
from mazelens.trainers import Trainer
from mazelens.util import compute_returns


class AlgorithmicDistillationTrainer(Trainer):
    env: RolloutEnvWrapper
    base_env: Env
    agent: Agent
    teacher_agent: Agent

    def __init__(self, device, seed, exp_dir, agent=None, env=None, epochs=None,
                 num_rollout_steps=None, eval_frequency=None,
                 num_environments=None, log_videos=None, teacher_agent=None):
        super().__init__(device, seed, exp_dir, agent, env, epochs, num_rollout_steps,
                         eval_frequency, num_environments, log_videos)
        self.teacher_agent_f = teacher_agent

    def init_train(self):
        super().init_train()
        self.teacher_agent = self.teacher_agent_f(action_space=self.base_env.action_space,
                                                  observation_space=self.base_env.observation_space,
                                                  device=self.device)

    def train(self):
        self.init_train()

        for epoch in tqdm(range(self.epochs)):
            with torch.no_grad():
                # generate rollouts from the teacher agent
                rollouts = self.env.rollout(agent=self.teacher_agent, num_steps=self.num_rollout_steps)

            # train the teacher agent
            self.teacher_agent.train(rollouts)
            # train the student agent
            self.agent.train(rollouts)

            if (epoch + 1) % self.eval_frequency == 0:
                stats = rollouts.compute_stats(0.99)
                print(f'Stats for epoch {epoch + 1}: {stats}')

                if self.log_videos:
                    rollouts.save_episode_to_mp4(os.path.join(self.exp_dir, 'videos', f'epoch_{epoch + 1}.mp4'))
