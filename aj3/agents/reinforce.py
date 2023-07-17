import torch

from aj3.agents.agent import Agent
from aj3.storage import RolloutStorage


class REINFORCEAgent(Agent):
    def update_policy(self, rollouts: RolloutStorage):
        batch = rollouts.to_batch()
        next_value = self.act(batch['states'][-1], hx=batch['hidden_states'][-1]).value_preds
        returns = rollouts.compute_returns(next_value, batch['rewards'], batch['done_mask'])

        loss = torch.flatten(-batch['log_probs'] * returns).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
