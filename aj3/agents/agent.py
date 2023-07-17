import abc
from dataclasses import dataclass, field

import torch
from torch import optim
from torch.distributions import Categorical


@dataclass
class PolicyOutput:
    action: torch.Tensor = field(default=None)
    log_prob: torch.Tensor = field(default=None)
    hidden_state: torch.Tensor = field(default=None)
    value_preds: torch.Tensor = field(default=None)
    features: torch.Tensor = field(default=None)


class Agent(abc.ABC):
    def __init__(self, env, network, cfg):
        self.env = env
        self.cfg = cfg
        self.policy = network
        self.optimizer = optim.Adam(self.policy.parameters(), lr=cfg.train.lr)

    def act(self, state, hx=None):
        state = state.long()[:, None]
        features, action_logits, hx = self.policy(state, hx=hx)
        features, action_logits = features.squeeze(1), action_logits.squeeze(1)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return PolicyOutput(action=action,
                            log_prob=log_prob.squeeze(-1),
                            features=features,
                            hidden_state=hx)

    @abc.abstractmethod
    def update_policy(self, rollouts):
        pass

    def to(self, device):
        self.policy.to(device)
        return self
