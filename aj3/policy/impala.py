import torch
from aj3.agents.agent import PolicyOutput
from einops import rearrange
from torch import nn

from aj3.util.configs import MazeArguments
from aj3.nets.blocks import ImpalaModel
from aj3.policy.policy import Policy


class ImpalaPolicyNet(Policy):
    def __init__(self, cfg: MazeArguments):
        super().__init__(cfg)
        self.cfg = cfg
        self.N = cfg.env.agent_visibility if cfg.env.agent_visibility != -1 else cfg.env.grid_size
        self.tok_embd = nn.Embedding(cfg.env.num_objects + 3, cfg.train.embd_dim)
        self.impala = ImpalaModel(cfg.train.embd_dim, self.N, cfg.train.hidden_size, cfg.train.scale)
        self.gru = nn.GRU(cfg.train.hidden_size, cfg.train.hidden_size, num_layers=cfg.train.gru_layers,
                          batch_first=True)

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

    def act(self, state=None, action=None, reward=None, hx=None):
        state, action, reward = map(lambda x: x[:, None] if x is not None else None, (state, action, reward))
        features, action_logits, hx = self.forward(state, hx=hx)
        action, log_prob = self.sample_action(action_logits)

        return PolicyOutput(action=action.squeeze(-1),
                            log_prob=log_prob.squeeze(-1),
                            features=features,
                            hidden_state=hx)
