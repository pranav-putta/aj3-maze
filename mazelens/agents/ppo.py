from dataclasses import dataclass

import torch
from einops import rearrange
from tensordict import TensorDict
from torch import nn
from torch.distributions import Categorical

from mazelens.agents.base_agent import Agent
from mazelens.agents.base_agent import AgentActionOutput, AgentInput
from mazelens.nets.base_net import Net
from mazelens.util import compute_returns
from mazelens.util.storage import RolloutStorage
import torch.nn.functional as F


@dataclass(kw_only=True)
class PPOAgentActionOutput(AgentActionOutput):
    values: torch.Tensor = None


class PPOAgent(Agent):
    policy: Net
    critic: Net

    def __init__(self, policy=None, critic=None, action_space=None, observation_space=None, deterministic=False,
                 epsilon=None, ppo_epochs=None, num_minibatches=None, value_loss_coef=None, entropy_coef=None,
                 max_grad_norm=None, lr=None, gamma=None, tau=None, use_gae=None, device=None):
        super().__init__(action_space, observation_space, deterministic)
        self.policy = policy
        self.critic = critic

        self.policy.to(device)
        self.critic.to(device)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        self.epsilon = epsilon
        self.ppo_epochs = ppo_epochs
        self.num_minibatches = num_minibatches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.gamma = gamma
        self.tau = tau
        self.use_gae = use_gae
        self.device = device

    def parameters(self):
        return list(self.policy.parameters()) + list(self.critic.parameters())

    def act(self, x: AgentInput) -> AgentActionOutput:
        features, action_logits, hx = self.policy(x, hx=x.prev_output.hiddens)
        dist = Categorical(logits=action_logits)
        if self.deterministic:
            actions = dist.mode
        else:
            actions = dist.sample()
        actions = actions
        log_probs = dist.log_prob(actions)
        values = self.critic(features).squeeze(-1, -2)
        return PPOAgentActionOutput(actions=actions, log_probs=log_probs, hiddens=hx, values=values)

    def evaluate_actions(self, states, actions):
        x = AgentInput(states=states, infos=None, prev_rewards=None, prev_output=None)
        feats, action_logits, _ = self.policy(x)
        values = self.critic(feats).squeeze(-1)
        dist = Categorical(logits=action_logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return values, log_probs, entropy

    def transform_input(self, x: AgentInput):
        x = super().transform_input(x)
        x.states = x.states.to(self.device)

        is_seq = len(x.states.shape) == 4
        if not is_seq:
            x.states = rearrange(x.states, 'b ... -> b 1 ...')

        return x

    def transform_output(self, states, rewards, infos, dones, successes, agent_output: PPOAgentActionOutput):
        return {
            'states': torch.from_numpy(states).long(),
            'rewards': torch.tensor(rewards).float(),
            'dones': torch.tensor(dones).bool(),
            'successes': torch.tensor(successes).bool(),
            'hiddens': agent_output.hiddens,
            'actions': agent_output.actions,
            'log_probs': agent_output.log_probs,
            'values': agent_output.values,
            'valids': torch.tensor(infos['valid'])
        }

    def learn(self, batch: TensorDict):
        batch['returns'] = compute_returns(rewards_t=batch['rewards'],
                                           done_mask_t=batch['dones'],
                                           values_t=batch['values'],
                                           gamma=self.gamma,
                                           tau=self.tau,
                                           use_gae=self.use_gae,
                                           next_value=None  # todo: use real next value and not fake
                                           )
        batch['returns'] = (batch['returns'] - batch['returns'].mean()) / (batch['returns'].std() + 1e-10)

        for _ in range(self.ppo_epochs):
            for minibatch in RolloutStorage.minibatch_generator(batch, self.num_minibatches):
                V, curr_log_probs, dist_entropy = self.evaluate_actions(minibatch['states'], minibatch['actions'])
                minibatch['returns'] = (minibatch['returns'] - minibatch['returns'].mean()) / (
                        minibatch['returns'].std() + 1e-10)
                A = minibatch['returns'] - minibatch['values']

                ratio = torch.exp(curr_log_probs - minibatch['log_probs'])
                surr1 = ratio * A
                surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * A
                entropy = dist_entropy.mean()

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = F.mse_loss(minibatch['returns'], V)

                total_loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                total_loss.backward()

                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def to(self, device):
        pass

    def save(self, path):
        pass

    def load(self, path):
        pass

    def initial_agent_output(self):
        return AgentActionOutput()
