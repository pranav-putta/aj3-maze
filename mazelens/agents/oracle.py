import torch

from mazelens.agents.base_agent import AgentActionOutput, AgentInput
from mazelens.agents.base_agent import Agent
from mazelens.util.storage import RolloutStorage


class OracleAgent(Agent):
    """ An agent that takes random actions. """

    def parameters(self):
        pass

    def act(self, x: AgentInput) -> AgentActionOutput:
        num_envs = len(x.states)
        actions = torch.tensor([a for a in x.infos['best_action']])
        log_probs = torch.tensor([0.0 for _ in range(num_envs)])
        return AgentActionOutput(actions=actions, log_probs=log_probs)

    def train(self, rollouts: RolloutStorage):
        pass

    def to(self, device):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
