import torch
from einops import rearrange
from torch import nn

from aj3.util.configs import MazeArguments
from aj3.policy.modules.focal_loss import FocalLoss
from aj3.policy.modules.gpt import GPT
from aj3.policy.policy import Policy


class DecisionTransformerPolicyNet(Policy):
    def __init__(self, cfg: MazeArguments):
        super().__init__()
        self.cfg = cfg

        self.N = cfg.env.agent_visibility if cfg.env.agent_visibility != -1 else cfg.env.grid_size
        self.tok_embd = nn.Embedding(cfg.env.num_objects + 3, cfg.train.embd_dim)
        self.convs = nn.Sequential(
            nn.Conv2d(cfg.train.embd_dim, cfg.train.embd_dim, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.projection = nn.Linear(cfg.train.embd_dim * self.N * self.N, cfg.train.hidden_size)

        self.gpt = GPT(
            block_size=1024,
            n_layer=2,
            n_head=4,
            n_embd=cfg.train.hidden_size,
            num_actions=4,
            obs_dim=cfg.train.hidden_size,
            mode='obs'
        )
        self.action_head = nn.Linear(cfg.train.hidden_size, 4)
        self.loss = FocalLoss(gamma=2, reduction='mean')

    def batch_into_episodes(self, batch):
        states, episode_lengths = [], []
        actions = []
        for env in range(self.cfg.train.num_envs):
            done_idxs = torch.where(batch['done_mask'][env])[0]
            starts = torch.cat([torch.tensor([0], device=self.cfg.device), done_idxs[:-1] + 1])
            ends = done_idxs
            for start, end in zip(starts, ends):
                states.append(batch['states'][env, start:end + 1])
                actions.append(batch['actions'][env, start:end + 1])
                episode_lengths.append(end - start + 1)
        states = torch.nn.utils.rnn.pad_sequence(states, batch_first=True)
        actions = torch.nn.utils.rnn.pad_sequence(actions, batch_first=True, padding_value=-100)

        return states, actions

    def compute_features(self, x, hx=None, return_pastkv=False):
        b, t, n, _ = x.shape
        x = x.to(self.cfg.device)

        # embed tokens
        x = rearrange(x, 'b t x y -> b t (x y)')
        x = self.tok_embd(x)
        x = rearrange(x, 'b t (x y) d -> (b t) d x y', x=n, y=n)

        # convolve
        x = self.convs(x)
        x = x.flatten(1)
        x = rearrange(x, '(b t) d -> b t d', b=b)

        # transformer
        x = self.projection(x)
        features = self.gpt(observations=x, past_kv=hx, return_past_kv=return_pastkv)
        return features

    def act(self, state, hx=None):
        features = self.compute_features(state, hx, return_pastkv=True)
        action_logits = self.action_head(features.logits)
        return features, action_logits

    def forward(self, batch, hx=None):
        x, actions = self.batch_into_episodes(batch)
        features = self.compute_features(x, hx=hx)
        # action head
        action_logits = self.action_head(features.logits)

        loss = self.loss(rearrange(action_logits, 'b t d -> (b t) d'), rearrange(actions, 'b t -> (b t)'))

        return features, action_logits, loss

    def initial_hidden_state(self):
        return self.gpt.empty_key_values(self.cfg.train.num_envs)
