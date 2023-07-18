import abc
from dataclasses import dataclass, field

import torch
from torch import optim
from torch.distributions import Categorical

from aj3.util.storage import RolloutStorage


@dataclass
class PolicyOutput:
    action: torch.Tensor = field(default=None)
    log_prob: torch.Tensor = field(default=None)
    hidden_state: torch.Tensor = field(default=None)
    value_preds: torch.Tensor = field(default=None)
    features: torch.Tensor = field(default=None)


class Agent(abc.ABC):
    def __init__(self, env, network, cfg):
        self.env = env
        self.cfg = cfg
        self.policy = network

    @abc.abstractmethod
    def update_policy(self, rollouts):
        pass

    def to(self, device):
        self.policy.to(device)
        return self

    def parameters(self):
        return self.policy.parameters()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def act(self, state=None, reward=None, action=None, hx=None):
        return self.policy.act(state=state, reward=reward, action=action, hx=hx)

    def collect_episodes(self, envs):
        num_steps = self.cfg.train.max_steps
        rollouts = RolloutStorage(self.cfg)

        curr_state = envs.reset()
        curr_rwd, curr_act = None, None
        hidden_state = self.policy.initial_hidden_state()
        for i in range(num_steps):
            curr_state = torch.tensor(curr_state, dtype=torch.long)
            policy_out = self.act(state=curr_state, reward=curr_rwd, action=curr_act, hx=hidden_state)
            acts = policy_out.action.cpu()
            next_states, rwds, dones, infos = envs.step(acts.cpu().numpy())

            success = [info['success'] for info in infos]
            valids = [info['valid'] for info in infos]

            rollouts.insert(curr_state, acts, policy_out.log_prob, rwds, dones, hidden_state, success, valids)

            curr_state = next_states
            curr_act = acts.clone()
            curr_rwd = torch.tensor(rwds, dtype=torch.float)
            hidden_state = policy_out.hidden_state
            if type(hidden_state) == torch.Tensor:
                hidden_state *= ~torch.tensor(dones, device=self.cfg.device).repeat(self.cfg.train.gru_layers,
                                                                                    self.cfg.train.hidden_size,
                                                                                    1).transpose(1, 2)

        # push the last state in the rollouts
        rollouts.states.append(torch.tensor(curr_state, dtype=torch.long))
        return rollouts
