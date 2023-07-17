import abc
from typing import Tuple

import torch


class Net(torch.nn.Module, abc.ABC):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def to(self, device):
        super().to(device)
        return self
