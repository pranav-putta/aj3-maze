import abc
from typing import Iterable, Any, Tuple

import numpy as np
import torch
from einops import rearrange
from torch.distributions import Categorical

from mazelens.util.storage import RolloutStorage
from mazelens.util.structs import ExperienceDict


class Agent(abc.ABC):
    """
    Abstract class for all agents.
    """

    def __init__(self, action_space=None, observation_space=None, deterministic=None, device=None, **kwargs):
        self.action_space = action_space
        self.observation_space = observation_space
        self.deterministic = deterministic
        self.device = device

    def sample_action(self, action_logits):
        dist = Categorical(logits=action_logits)
        if self.deterministic:
            actions = dist.mode
        else:
            actions = dist.sample()
        actions = actions.squeeze(-1)
        log_probs = dist.log_prob(actions)
        return actions, log_probs

    @abc.abstractmethod
    def act(self, x: ExperienceDict) -> Tuple[Iterable[int], Any]:
        """
        Act on a batch of states.

        @param x:
        @return: (actions, hidden)
        """
        pass

    @abc.abstractmethod
    def update(self, rollouts: RolloutStorage):
        """
        Update the agent's internal state given a transition.
        """
        pass

    @abc.abstractmethod
    def save(self, path):
        """
        Save the agent's state to a file.
        """
        pass

    @abc.abstractmethod
    def load(self, path):
        """
        Load the agent's state from a file.
        """
        pass

    @abc.abstractmethod
    def to(self, device):
        """
        Move the agent to a device.
        """
        pass

    def before_step(self, x: ExperienceDict):
        """
        convert states, rewards, infos to a batch
        """
        x.states = torch.from_numpy(np.array(x.states)).to(self.device)
        is_seq = len(x.states.shape) == 4
        if not is_seq:
            x.states = rearrange(x.states, 'b ... -> b 1 ...')

        x.prev_dones = torch.from_numpy(np.array(x.prev_dones)).to(self.device)

        if x.actions is not None and not is_seq:
            x.actions = torch.from_numpy(np.array(x.actions.cpu())).to(self.device)
            x.actions = rearrange(x.actions, 'b ... -> b 1 ...')
        if x.rewards is not None:
            x.rewards = torch.from_numpy(np.array(x.rewards)).to(self.device)
            x.rewards = rearrange(x.rewards, 'b ... -> b 1 ...')
        if x.prev_hiddens is not None and type(x.prev_hiddens) == torch.Tensor:
            x.prev_hiddens = x.prev_hiddens.to(self.device)

        return x

    def after_step(self, x: ExperienceDict):
        """
        convert output to a batch
        """
        x.rewards = torch.tensor(x.rewards, dtype=torch.float, device=self.device)
        x.states = x.states.squeeze(1)
        x.truncated = torch.tensor(x.truncated, dtype=torch.bool, device=self.device)
        x.success = torch.tensor(x.success, dtype=torch.bool, device=self.device)
        return x

    def initialize_hidden(self, batch_size):
        return None

    @abc.abstractmethod
    def parameters(self):
        pass
