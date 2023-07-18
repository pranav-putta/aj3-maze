import abc
from typing import Tuple

import torch

from aj3.util.configs import MazeArguments


class Policy(torch.nn.Module, abc.ABC):
    def __init__(self, cfg: MazeArguments):
        super().__init__()
        self.cfg = cfg

    def forward(self, x) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pass

    def to(self, device):
        super().to(device)
        return self

    def act(self):
        pass

    def sample_action(self, action_logits):
        dist = torch.distributions.Categorical(logits=action_logits)
        act = None
        if self.cfg.train.deterministic:
            act = dist.mode
        else:
            act = dist.sample()

        return act, dist.log_prob(act)

    def initial_hidden_state(self):
        cfg = self.cfg
        return torch.zeros((cfg.train.gru_layers, cfg.train.num_envs, cfg.train.hidden_size),
                           dtype=torch.float,
                           device=cfg.device)

    def transform_batch(self, batch):
        return batch
