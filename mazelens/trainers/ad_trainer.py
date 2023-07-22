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
import wandb


class AlgorithmicDistillationTrainer(Trainer):
    env: RolloutEnvWrapper
    base_env: Env
    agent: Agent
    teacher_agent: Agent

    def __init__(self, teacher_agent=None, **kwargs):
        super().__init__(**kwargs)
        self.teacher_agent_f = teacher_agent

    def init_train(self):
        super().init_train()
        self.teacher_agent = self.teacher_agent_f(action_space=self.base_env.action_space,
                                                  observation_space=self.base_env.observation_space,
                                                  device=self.device)
        self.teacher_agent.to(self.device)

        if self.teacher_agent.parameters() is not None:
            print("Agent parameters: ", sum(p.numel() for p in self.teacher_agent.parameters() if p.requires_grad))
        else:
            print("Agent does not have parameters")

    def train(self):
        self.init_train()

        for epoch in tqdm(range(self.epochs)):
            with torch.no_grad():
                # generate rollouts from the teacher agent
                rollouts = self.env.rollout(agent=self.teacher_agent, num_steps=self.num_rollout_steps)

            # train the teacher agent
            teacher_loss = self.teacher_agent.update(rollouts)
            loss = self.agent.update(rollouts)

            self.logger.log({"teacher_loss": teacher_loss,
                             "student_loss": loss})

            if (epoch + 1) % self.eval_frequency == 0:
                teacher_stats = rollouts.compute_stats(0.99)
                student_rollout = self.env.rollout(agent=self.agent, num_steps=self.num_rollout_steps)
                student_stats = student_rollout.compute_stats(0.99)

                if self.log_videos:
                    rollouts.save_episode_to_mp4(
                        os.path.join(self.exp_dir, 'videos', f'epoch_{epoch + 1}_teacher.mp4'))
                    student_rollout.save_episode_to_mp4(
                        os.path.join(self.exp_dir, 'videos', f'epoch_{epoch + 1}_student.mp4'))

                self.agent.save(
                    os.path.join(self.exp_dir, 'checkpoints', f'epoch_{epoch + 1}_student.pt'))
                self.teacher_agent.save(
                    os.path.join(self.exp_dir, 'checkpoints', f'epoch_{epoch + 1}_teacher.pt'))

                self.logger.log({"teacher_success_rate": teacher_stats.success_rate,
                                 "teacher_mean_returns": teacher_stats.avg_reward,
                                 "student_success_rate": student_stats.success_rate,
                                 "student_mean_returns": student_stats.avg_reward})
