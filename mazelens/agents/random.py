import torch

from mazelens.agents.base_agent import AgentActionOutput, AgentInput
from mazelens.agents.base_agent import Agent
from mazelens.util.storage import RolloutStorage


class RandomAgent(Agent):
    """ An agent that takes random actions. """

    def act(self, x: AgentInput) -> AgentActionOutput:
        num_envs = len(x.states)
        action = torch.tensor([self.action_space.sample() for _ in range(num_envs)])
        log_probs = torch.tensor([0.0 for _ in range(num_envs)])
        return AgentActionOutput(actions=action, log_probs=log_probs)

    def train(self, rollouts: RolloutStorage):
        pass

    def to(self, device):
        pass

    def train(self, obs, action, reward, next_obs, done):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
