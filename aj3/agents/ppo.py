import gym
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical
from torch.distributions import MultivariateNormal

from aj3.agents.agent import Agent


class PPOAgent(Agent):
    def __init__(self, env, network, epsilon=0.2, value_coef=0.5, entropy_coef=0.01):
        super().__init__(env)
        self.env = env
        self.net = network
        self.epsilon = epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def act(self, state):
        state = torch.from_numpy(state).long()
        logits, value = self.net(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action.item(), log_prob, value

    def update_policy(self, states, actions, log_probs, returns, advantages):
        states = torch.tensor(np.array(states)).long()
        actions = torch.tensor(np.array(actions)).long()
        log_probs = torch.tensor(log_probs).float()
        returns = torch.tensor(returns).float()
        advantages = torch.tensor(advantages).float()

        logits, values = self.net(states)
        dist = Categorical(logits=logits)
        new_log_probs = dist.log_prob(actions)
        ratio = torch.exp(new_log_probs - log_probs)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()
        critic_loss = (returns - values).pow(2).mean()

        entropy = dist.entropy().mean()

        total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
