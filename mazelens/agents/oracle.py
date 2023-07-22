from typing import Tuple, Iterable, Any

import torch

from mazelens.agents.base_agent import Agent
from mazelens.util.storage import RolloutStorage
from mazelens.util.structs import ExperienceDict


class OracleAgent(Agent):
    """ An agent that takes random actions. """

    def parameters(self):
        pass

    def act(self, x: ExperienceDict) -> Tuple[Iterable[int], Any]:
        num_envs = len(x.states)
        actions = torch.tensor([a for a in x.infos['best_action']])
        return actions, None

    def update(self, rollouts: RolloutStorage):
        pass

    def to(self, device):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass
