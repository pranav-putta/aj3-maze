import abc

import torch

from mazelens.agents import AgentInput


class Net(torch.nn.Module, abc.ABC):
    """ Abstract class for all policies. """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
