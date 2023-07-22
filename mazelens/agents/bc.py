from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import torch
from einops import rearrange
from torch import nn
from torch.distributions import Categorical

from mazelens.agents.base_agent import Agent
from mazelens.nets.base_net import Net
from mazelens.nets.modules.focal_loss import FocalLoss
from mazelens.util.storage import RolloutStorage
from mazelens.util.structs import ExperienceDict


class BCAgent(Agent):
    policy: Net

    def __init__(self, lr, max_grad_norm, device, policy, **kwargs):
        super().__init__(**kwargs)
        self.policy = policy

        self.policy.to(device)

        self.lr = lr
        self.max_grad_norm = max_grad_norm
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.criterion = FocalLoss(gamma=2.0, ignore_index=4)

    def parameters(self):
        return list(self.policy.parameters())

    def before_step(self, x: ExperienceDict):
        x = super().before_step(x)
        return x

    def after_step(self, x: ExperienceDict):
        x = super().after_step(x)
        return x

    def initialize_hidden(self, batch_size):
        if hasattr(self.policy, 'rnn'):
            return self.policy.rnn.initialize_hidden(batch_size).to(self.device)

    def act(self, x: ExperienceDict) -> Tuple[Iterable[int], Any]:
        features, action_logits, hx = self.policy(x)
        actions, log_probs = self.sample_action(action_logits)
        return actions, hx

    def update(self, rollouts: RolloutStorage):
        batch = rollouts.to_tensordict()
        batch = batch.to(self.device)

        # construct experience dict
        agent_input = ExperienceDict(prev_hiddens=batch['prev_hiddens'],
                                     prev_dones=batch['prev_dones'],
                                     states=batch['states'],
                                     actions=batch['actions'],
                                     rewards=batch['rewards'])
        features, action_logits, hx = self.policy(agent_input)
        actions = agent_input.prev_state.actions

        # we don't have to shift actions because actions describes the action taken at the next step already
        X = rearrange(action_logits, 'b t d -> (b t) d')
        Y = rearrange(actions, 'b t -> (b t)')

        loss = self.criterion(X, Y)

        self.optimizer.zero_grad()
        loss.backward()

        nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
        self.optimizer.step()
        return loss.item()

    def to(self, device):
        self.policy.to(device)

    def save(self, path):
        torch.save({'policy': self.policy.state_dict(), 'optimizer': self.optimizer.state_dict()}, path)

    def load(self, path):
        pass
