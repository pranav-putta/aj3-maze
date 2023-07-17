from dataclasses import dataclass

import numpy as np
import torch
from einops import rearrange
from tensordict import TensorDict

from aj3.configs import MazeArguments


@dataclass
class RolloutStats:
    success_rate: float
    num_episodes: int

    avg_episode_length: float
    avg_reward: float


class RolloutStorage:
    def __init__(self, cfg: MazeArguments):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.done_mask = []
        self.hidden_states = []
        self.successes = []

        self.cfg = cfg
        self.current_step = 0

    def insert(self, state, action, log_prob, reward, done, hidden_state, success):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(list(reward))
        self.done_mask.append(list(done))
        self.hidden_states.append(hidden_state)
        self.successes.append(success)

        self.current_step += 1

    def compute_returns(self, next_value=None, rewards_t=None, done_mask_t=None):
        if next_value is None:
            next_value = torch.zeros(self.cfg.train.num_envs, device=self.cfg.device)
        returns = [torch.zeros(self.cfg.train.num_envs, 1) for _ in range(self.current_step + 1)]
        returns[self.current_step] = next_value
        for step in reversed(range(self.current_step)):
            returns[step] = (rewards_t[:, step]
                             + self.cfg.train.gamma
                             * returns[step + 1]
                             * (~done_mask_t[:, step]))
        return torch.stack(returns[:-1]).transpose(0, 1)

    def to_batch(self):
        batch = TensorDict({'states': torch.from_numpy(np.array(self.states[:-1])).long(),
                            'actions': torch.stack(self.actions),
                            'log_probs': torch.stack(self.log_probs),
                            'rewards': torch.tensor(self.rewards),
                            'done_mask': torch.tensor(self.done_mask),
                            'hidden_states': torch.stack(self.hidden_states),
                            'successes': torch.tensor(self.successes)},
                           batch_size=[self.cfg.train.max_steps])
        batch = batch.apply(lambda v: v.to(self.cfg.device))
        batch = batch.apply(lambda v: rearrange(v, 't b ... -> b t ...'))
        return batch

    def compute_stats(self, batch):
        returns = self.compute_returns(None, batch['rewards'], batch['done_mask'])

        num_episodes = 0
        avg_returns = 0
        avg_episode_lengths = 0

        for env in range(self.cfg.train.num_envs):
            done_idxs = torch.where(batch['done_mask'][env])[0]
            starts = torch.cat([torch.tensor([0], device=self.cfg.device), done_idxs[:-1] + 1])
            ends = done_idxs
            avg_episode_lengths += (ends - starts).sum().item()
            avg_returns += returns[env, starts].sum().item()
            num_episodes += len(starts)

        avg_episode_length = avg_episode_lengths / num_episodes
        avg_returns = avg_returns / num_episodes
        sr = (batch['successes'].sum() / num_episodes).item()

        return RolloutStats(success_rate=sr,
                            num_episodes=num_episodes,
                            avg_episode_length=avg_episode_length,
                            avg_reward=avg_returns)
