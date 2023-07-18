import torch
import torch.nn.functional as F
from einops import rearrange

from aj3.util.configs import MazeArguments
from torch import nn
from torch import optim
from torch.distributions import Categorical
from tensordict import TensorDict

from aj3.agents.agent import Agent
from aj3.util.storage import RolloutStorage


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

        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.critic.parameters()))

    def save(self, path):
        torch.save({'policy': self.policy.state_dict(),
                    'critic': self.critic.state_dict(),
                    'optim': self.optimizer.state_dict()}, path)

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

    def minibatch_generator(self, states, actions, old_values, returns, log_probs):
        new_batch_size = self.cfg.train.num_envs // self.cfg.train.num_minibatches
        for inds in torch.randperm(self.cfg.train.num_envs).chunk(self.cfg.train.num_minibatches):
            yield TensorDict({'states': states[inds],
                              'actions': actions[inds],
                              'values': old_values[inds],
                              'returns': returns[inds],
                              'log_probs': log_probs[inds]}, batch_size=[new_batch_size], )

    def update_policy(self, rollouts: RolloutStorage):
        batch = rollouts.to_batch()

        with torch.no_grad():
            old_values, old_log_probs, _ = self.evaluate_actions(batch['states'], batch['actions'])
            next_value = self.evaluate_actions(rollouts.states[-1][:, None], None).squeeze(-1)

        returns = rollouts.compute_returns(next_value, batch['rewards'], batch['done_mask'], value_preds=old_values)
        # returns = (returns - returns.mean()) / (returns.std() + 1e-10)
        eps = self.cfg.train.epsilon

        for _ in range(self.cfg.train.ppo_epochs):
            for minibatch in self.minibatch_generator(batch['states'],
                                                      batch['actions'],
                                                      old_values,
                                                      returns, old_log_probs):
                V, curr_log_probs, dist_entropy = self.evaluate_actions(minibatch['states'], minibatch['actions'])
                minibatch['returns'] = (minibatch['returns'] - minibatch['returns'].mean()) / (
                        minibatch['returns'].std() + 1e-10)
                A = minibatch['returns'] - minibatch['values']

                ratio = torch.exp(curr_log_probs - minibatch['log_probs'])
                surr1 = ratio * A
                surr2 = torch.clamp(ratio, 1 - eps, 1 + eps) * A
                entropy = dist_entropy.mean()

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = F.mse_loss(minibatch['returns'], V)

                total_loss = actor_loss + self.cfg.train.value_coef * critic_loss - self.cfg.train.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), self.cfg.train.max_grad_norm)
                self.optimizer.step()

    def parameters(self):
        return list(self.policy.parameters()) + list(self.critic.parameters())
