import abc
from dataclasses import dataclass

import numpy as np
import torch
from tensordict import TensorDict

from mazelens.util.storage import RolloutStorage


@dataclass(kw_only=True)
class AgentActionOutput:
    actions: torch.Tensor = None
    log_probs: torch.Tensor = None
    hiddens: torch.Tensor = None


@dataclass
class AgentInput:
    states: torch.Tensor | None
    infos: TensorDict | None
    prev_rewards: torch.Tensor | None
    prev_output: AgentActionOutput | None


class Agent(abc.ABC):
    """
    Abstract class for all agents.
    """

    def __init__(self, action_space=None, observation_space=None, deterministic=None):
        self.action_space = action_space
        self.observation_space = observation_space
        self.deterministic = deterministic

    @abc.abstractmethod
    def act(self, x: AgentInput) -> AgentActionOutput:
        """
        Return an action given an observation.
        """
        pass

    @abc.abstractmethod
    def learn(self, rollouts: RolloutStorage):
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

    def transform_input(self, x: AgentInput):
        """
        convert states, rewards, infos to a batch
        """
        x.states = torch.from_numpy(np.array(x.states))
        if x.prev_output.actions is not None:
            x.prev_output.actions = torch.from_numpy(np.array(x.prev_output.actions))
        if x.prev_rewards is not None:
            x.prev_rewards = torch.from_numpy(np.array(x.prev_rewards))
        return x

    def transform_output(self, states, rewards, infos, dones, successes, agent_output: AgentActionOutput):
        """
        convert output to a batch
        """
        return {'states': torch.from_numpy(np.array(states)),
                'rewards': torch.from_numpy(rewards),
                'valids': torch.tensor(infos['valid']),
                'actions': agent_output.actions,
                'dones': torch.from_numpy(dones),
                'successes': torch.from_numpy(successes),
                'log_probs': agent_output.log_probs}

    def initial_agent_output(self):
        return AgentActionOutput()
