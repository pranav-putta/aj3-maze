import torch
import torch.nn.functional as F
from einops import rearrange

from aj3.configs import MazeArguments
from torch import nn
from torch import optim
from torch.distributions import Categorical
from tensordict import TensorDict

from aj3.agents.agent import Agent
from aj3.storage import RolloutStorage


class CriticHead(nn.Module):
    def __init__(self, cfg: MazeArguments, input_size):
        super().__init__()
        self.cfg = cfg
        self.fc = nn.Linear(input_size, 1)
        nn.init.orthogonal_(self.fc.weight)
        nn.init.constant_(self.fc.bias, 0)

    def forward(self, x):
        x = x.to(self.cfg.device)
        return self.fc(x)


class PPOAgent(Agent):

    def __init__(self, env, network, cfg: MazeArguments):
        super().__init__(env, network, cfg)
        self.critic = CriticHead(cfg, cfg.train.hidden_size)

    def to(self, device):
        self.critic.to(device)
        return super().to(device)

    def evaluate_actions(self, states, actions=None):
        feats, action_logits, _ = self.policy(states)
        values = self.critic(feats).squeeze(-1)
        if actions is not None:
            dist = Categorical(logits=action_logits)
            log_probs = dist.log_prob(actions)
            entropy = dist.entropy()
            return values, log_probs, entropy
        else:
            return values

    def update_policy(self, rollouts: RolloutStorage):
        batch = rollouts.to_batch()

        with torch.no_grad():
            V, old_log_probs, _ = self.evaluate_actions(batch['states'], batch['actions'])
            next_value = self.evaluate_actions(rollouts.states[-1][:, None], None).squeeze(-1)

        returns = rollouts.compute_returns(next_value, batch['rewards'], batch['done_mask'])
        returns = (returns - returns.mean()) / (returns.std() + 1e-10)
        A_k = returns - V

        for _ in range(self.cfg.train.ppo_epochs):
            V, curr_log_probs, dist_entropy = self.evaluate_actions(batch['states'], batch['actions'])

            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio * A_k
            surr2 = torch.clamp(ratio, 1 - self.cfg.train.epsilon, 1 + self.cfg.train.epsilon) * A_k
            entropy = dist_entropy.mean()

            actor_loss = (-torch.min(surr1, surr2)).mean()
            critic_loss = F.mse_loss(returns, V)

            total_loss = actor_loss + self.cfg.train.value_coef * critic_loss - self.cfg.train.entropy_coef * entropy

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()
