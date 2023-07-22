from dataclasses import dataclass
from typing import List, Any

import torch
from tensordict import TensorDict

from mazelens.util.storage import RolloutStorage


@dataclass
class ExperienceDict:
    prev_dones: torch.Tensor | List[bool]
    prev_hiddens: torch.Tensor | List[Any]
    states: torch.Tensor
    infos: Any
    actions: torch.Tensor | None
    rewards: torch.Tensor | None
    success: torch.Tensor | None
    truncated: torch.Tensor | None

    def __init__(self, prev_dones=None, prev_hiddens=None, states=None,
                 infos=None, actions=None, rewards=None, success=None, truncated=None):
        self.prev_dones = prev_dones
        self.prev_hiddens = prev_hiddens
        self.states = states
        self.infos = infos
        self.actions = actions
        self.rewards = rewards
        self.success = success
        self.truncated = truncated
