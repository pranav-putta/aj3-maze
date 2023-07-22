from dataclasses import dataclass
from typing import Tuple, Iterable, Any

import torch
import torch.nn.functional as F
from einops import rearrange
from tensordict import TensorDict
from torch import nn
from torch.distributions import Categorical

from mazelens.agents.base_agent import Agent
from mazelens.nets.base_net import Net
from mazelens.util import compute_returns
from mazelens.util.storage import RolloutStorage
from mazelens.util.structs import ExperienceDict


class PPOAgent(Agent):
    policy: Net
    critic: Net

    def __init__(self, policy=None, critic=None, action_space=None, observation_space=None, deterministic=False,
                 epsilon=None, ppo_epochs=None, num_minibatches=None, val_loss_coef=None, entropy_coef=None,
                 max_grad_norm=None, lr=None, gamma=None, tau=None, use_gae=None, device=None):
        super().__init__(action_space, observation_space, deterministic, device)
        self.policy = policy
        self.critic = critic

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.epsilon = epsilon
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.value_loss_coef = val_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.tau = tau
        self.use_gae = use_gae

    def parameters(self):
        return list(self.policy.parameters()) + list(self.critic.parameters())

    def before_step(self, x: ExperienceDict):
        return super().before_step(x)

    def after_step(self, x: ExperienceDict):
        x = super().after_step(x)
        return x

    def initialize_hidden(self, batch_size):
        if hasattr(self.policy, 'rnn'):
            return self.policy.rnn.initialize_hidden(batch_size).to(self.device)

    def act(self, x: ExperienceDict) -> Tuple[Iterable[int], Any]:
        self.policy.eval()
        features, action_logits, hx = self.policy(x)
        actions, log_probs = self.sample_action(action_logits)

        return actions, hx

    def evaluate_actions(self, prev_hiddens, prev_dones, states, actions):
        # since we're feeding in the whole sequence, don't pass previous hidden state
        exp = ExperienceDict(prev_hiddens=prev_hiddens, prev_dones=prev_dones,
                             states=states, actions=actions, rewards=None, infos=None,
                             truncated=None, success=None)
        feats, action_logits, _ = self.policy(exp)

        values = self.critic(feats).squeeze(-1, -2)

        # compute log probs and entropy
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()

        return values, log_probs, entropy

    @torch.no_grad()
    def compute_next_value(self, exp: ExperienceDict):
        exp = self.before_step(exp)
        feats, _, _ = self.policy(exp)
        value = self.critic(feats).squeeze(-1, -2)
        return value

    def update(self, rollouts: RolloutStorage):
        self.policy.train()
        batch = rollouts.to_tensordict()
        batch = batch.apply(lambda x: x.to(self.device))

        with torch.no_grad():
            next_value = self.compute_next_value(rollouts.last_exp)
            values, old_log_probs, _ = self.evaluate_actions(batch['prev_hiddens'],
                                                             batch['prev_dones'],
                                                             batch['states'],
                                                             batch['actions'])
        batch['old_log_probs'] = old_log_probs
        batch['values'] = values
        batch['returns'] = compute_returns(rewards_t=batch['rewards'],
                                           done_mask_t=batch['prev_dones'],
                                           values_t=values,
                                           gamma=self.gamma,
                                           tau=self.tau,
                                           use_gae=self.use_gae,
                                           next_value=next_value)
        batch['returns'] = (batch['returns'] - batch['returns'].mean()) / (batch['returns'].std() + 1e-5)

        avg_actor_loss = 0.
        avg_critic_loss = 0.
        for _ in range(self.ppo_epochs):
            for mb in RolloutStorage.minibatch_generator(batch, self.num_minibatches):
                V, curr_log_probs, dist_entropy = self.evaluate_actions(mb['prev_hiddens'],
                                                                        mb['prev_dones'],
                                                                        mb['states'],
                                                                        mb['actions'])
                # normalize returns
                A = mb['returns'] - mb['values']

                ratio = torch.exp(curr_log_probs - mb['old_log_probs'])
                surr1 = ratio * A
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A
                entropy = dist_entropy.mean()

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = F.mse_loss(mb['returns'], V)

                total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

                avg_actor_loss += actor_loss.item()
                avg_critic_loss += critic_loss.item()

        avg_actor_loss /= self.ppo_epochs * self.num_minibatches
        avg_critic_loss /= self.ppo_epochs * self.num_minibatches
        return {
            'actor_loss': avg_actor_loss,
            'critic_loss': avg_critic_loss,
        }

    def to(self, device):
        self.policy.to(device)
        self.critic.to(device)

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, path)

    def load(self, path):
        pass
