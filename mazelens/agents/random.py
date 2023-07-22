import torch

from mazelens.agents.base_agent import Agent
from mazelens.util.storage import RolloutStorage
from mazelens.util.structs import ExperienceDict


class RandomAgent(Agent):
    """ An agent that takes random actions. """

    def act(self, x: ExperienceDict):
        num_envs = len(x.states)
        actions = torch.tensor([self.action_space.sample() for _ in range(num_envs)])
        return actions, None

    def update(self, rollouts: RolloutStorage):
        pass

    def to(self, device):
        pass

    def train(self, obs, action, reward, next_obs, done):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
