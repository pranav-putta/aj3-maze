import torch
from einops import rearrange
from torch import nn

from aj3.configs import MazeArguments
from aj3.nets.blocks import ImpalaModel
from aj3.nets.net import Net


class ImpalaPolicyNet(Net):
    def __init__(self, cfg: MazeArguments):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.env.agent_visibility if cfg.env.agent_visibility != -1 else cfg.env.grid_size
        self.tok_embd = nn.Embedding(cfg.env.num_objects + 3, cfg.train.embd_dim)
        self.impala = ImpalaModel(cfg.train.embd_dim, self.N, cfg.train.hidden_size)
        self.gru = nn.GRU(cfg.train.embd_dim, cfg.train.hidden_size, num_layers=cfg.train.gru_layers, batch_first=True)

        self.action_head = nn.Linear(cfg.train.hidden_size, 4)

    def forward(self, x, hx=None):
        b, t, n, _ = x.shape
        x = x.to(self.cfg.device)

        # embed tokens
        x = rearrange(x, 'b t x y -> b t (x y)')
        x = self.tok_embd(x)
        x = rearrange(x, 'b t (x y) d -> (b t) d x y', x=n, y=n)

        # impala + gru
        x = self.impala(x)
        x = rearrange(x, '(b t) d -> b t d', b=b)
        feats, hx = self.gru(x, hx)

        action_logits = self.action_head(feats)
        return feats, action_logits, hx
