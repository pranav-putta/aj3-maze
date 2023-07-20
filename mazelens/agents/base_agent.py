import abc
from dataclasses import dataclass

import numpy as np
import torch
from tensordict import TensorDict

from mazelens.util import shift_tensor_sequence
from mazelens.util.storage import RolloutStorage


@dataclass(kw_only=True)
class AgentActionOutput:
    actions: torch.Tensor | None = None
    log_probs: torch.Tensor | None = None
    hiddens: torch.Tensor | None = None
    rewards: torch.Tensor | None = None


@dataclass
class AgentInput:
    states: torch.Tensor | None
    infos: TensorDict | None
    prev: AgentActionOutput | None


class Agent(abc.ABC):
    """
    Abstract class for all agents.
    """

    def __init__(self, action_space=None, observation_space=None, deterministic=None, **kwargs):
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
    def train(self, rollouts: RolloutStorage):
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
        if x.prev.actions is not None:
            x.prev.actions = x.prev.actions.clone()
        if x.prev.rewards is not None:
            x.prev.rewards = torch.from_numpy(np.array(x.prev.rewards))
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

    @abc.abstractmethod
    def parameters(self):
        pass

    @staticmethod
    def construct_policy_input(states, actions=None, hiddens=None, log_probs=None, rewards=None,
                               shift=False):
        if actions is not None and shift:
            actions = shift_tensor_sequence(actions, 0, dim=0)
        if hiddens is not None and shift:
            hiddens = shift_tensor_sequence(hiddens, 0, dim=0)
        if log_probs is not None and shift:
            log_probs = shift_tensor_sequence(log_probs, 0, dim=0)
        if rewards is not None and shift:
            rewards = shift_tensor_sequence(rewards, 0, dim=0)

        prev_output = AgentActionOutput(actions=actions,
                                        hiddens=hiddens,
                                        log_probs=log_probs,
                                        rewards=rewards)
        agent_input = AgentInput(states=states,
                                 prev=prev_output,
                                 infos=None)
        return agent_input
