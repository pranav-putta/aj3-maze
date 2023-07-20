from dataclasses import dataclass

import torch
from einops import rearrange
from tensordict import TensorDict
from torch import nn
from torch.distributions import Categorical

from mazelens.agents.base_agent import Agent
from mazelens.agents.base_agent import AgentActionOutput, AgentInput
from mazelens.nets.base_net import Net
from mazelens.nets.modules.focal_loss import FocalLoss
from mazelens.util import compute_returns
from mazelens.util.storage import RolloutStorage
import torch.nn.functional as F


@dataclass(kw_only=True)
class PPOAgentActionOutput(AgentActionOutput):
    values: torch.Tensor = None


class BCAgent(Agent):
    policy: Net

    def __init__(self, policy=None, action_space=None, observation_space=None, deterministic=False,
                 device=None, lr=None):
        super().__init__(action_space, observation_space, deterministic)
        self.policy = policy

        self.policy.to(device)

        self.lr = lr
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.device = device
        self.criterion = FocalLoss(gamma=2.0)

    def parameters(self):
        return list(self.policy.parameters())

    def act(self, x: AgentInput) -> AgentActionOutput:
        features, action_logits, hx = self.policy(x, hx=x.prev.hiddens)
        dist = Categorical(logits=action_logits)
        if self.deterministic:
            actions = dist.mode
        else:
            actions = dist.sample()
        actions = actions.squeeze(-1)
        log_probs = dist.log_prob(actions)
        return AgentActionOutput(actions=actions, log_probs=log_probs, hiddens=hx)

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

    def train(self, rollouts: RolloutStorage):
        batch = rollouts.to_tensordict()

        # Forward through policy network
        policy_input = Agent.construct_policy_input(states=batch['states'],
                                                    actions=batch['actions'],
                                                    rewards=batch['rewards'])
        features, action_logits, hx = self.policy(policy_input)

        F.mse_loss(action_logits[:-1], batch['actions'][1:])

    def to(self, device):
        self.policy.to(device)

    def save(self, path):
        pass

    def load(self, path):
        pass

    def initial_agent_output(self):
        return AgentActionOutput()
