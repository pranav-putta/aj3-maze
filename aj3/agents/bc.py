import torch
from torch.distributions import Categorical

from aj3.agents.agent import Agent, PolicyOutput
from aj3.util.storage import RolloutStorage


class BCAgent(Agent):
    def __init__(self, env, network, cfg):
        super().__init__(env, network, cfg)
        self.optimizer = torch.optim.Adam(self.policy.parameters())

    def update_policy(self, rollouts: RolloutStorage):
        batch = self.policy.transform_batch(rollouts.to_batch())
        feats, logits, loss = self.policy(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, path):
        torch.save({'policy': self.policy.state_dict(),
                    'optimizer': self.optimizer.state_dict()}, path)
