import abc

import torch
from tensordict import TensorDict

from mazelens.agents import AgentInput


class Net(torch.nn.Module, abc.ABC):
    """ Abstract class for all policies. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def transform_batch_to_input(self, batch: TensorDict):
        """ Transform a batch of observations into a batch of states. """
        raise NotImplementedError

    def initial_hidden_state(self, batch_size):
        raise NotImplementedError

    def forward(self, x, generation_mode=False):
        """ Forward pass of the network. """
        raise NotImplementedError
