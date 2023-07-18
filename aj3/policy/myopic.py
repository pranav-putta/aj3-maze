import torch
from einops import rearrange
from torch import nn

from aj3.util.configs import MazeArguments
from aj3.policy.policy import Policy


class MyopicPolicyNet(Policy):
    def __init__(self, cfg: MazeArguments):
        super().__init__()
        self.cfg = cfg

        self.N = cfg.env.agent_visibility if cfg.env.agent_visibility != -1 else cfg.env.grid_size
        self.tok_embd = nn.Embedding(cfg.env.num_objects + 3, cfg.train.embd_dim)

        self.fc = nn.Sequential(nn.Linear(cfg.train.embd_dim * self.N * self.N, cfg.train.hidden_size),
                                nn.ReLU(),
                                nn.Linear(cfg.train.hidden_size, cfg.train.hidden_size))

        self.action_head = nn.Linear(cfg.train.hidden_size, 4)

    def forward(self, x, hx=None):
        b, t, n, _ = x.shape
        x = x.to(self.cfg.device)

        # embed tokens
        x = rearrange(x, 'b t x y -> b t (x y)')
        x = self.tok_embd(x)

        # action head
        x = x.flatten(2)
        feats = self.fc(x)

        action_logits = self.action_head(feats)

        return feats, action_logits, hx
