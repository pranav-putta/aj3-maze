import gym
import numpy as np
import torch
from torch import nn
from torch import optim
from torch.distributions import Categorical

from aj3.agents.agent import Agent


class REINFORCEAgent(Agent):
    def __init__(self, env, network):
        super().__init__(env)
        self.env = env
        self.net = network
        self.optimizer = optim.Adam(self.net.parameters(), lr=1e-3)

    def act(self, state):
        state = torch.from_numpy(state).long()
        logits = self.net(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        self.net(state)
        self.last_state = state
        return action.item(), dist.log_prob(action)

    def update_policy(self, rewards, log_probs):
        returns = []
        discounted_reward = 0
        for r in reversed(rewards):
            discounted_reward = r + discounted_reward * 0.99  # Discount factor (gamma)
            returns.insert(0, discounted_reward)
        returns = torch.tensor(returns)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)  # Normalize the returns

        loss = []
        for log_prob, R in zip(log_probs, returns):
            loss.append(-log_prob * R)

        self.optimizer.zero_grad()
        loss = torch.cat(loss).sum()
        loss.backward()
        self.optimizer.step()

        try:
            if torch.isnan(self.net.tok_embd.weight).any():
                self.net(self.last_state)
        except:
            self.net(self.last_state)
