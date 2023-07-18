from einops import rearrange
from torch import nn

from aj3.agents.agent import PolicyOutput
from aj3.policy.policy import Policy
from aj3.util.configs import MazeArguments


class SimplePolicyNet(Policy):
    def __init__(self, cfg: MazeArguments):
        super().__init__(cfg)
        self.cfg = cfg

        self.N = cfg.env.agent_visibility if cfg.env.agent_visibility != -1 else cfg.env.grid_size

        self.tok_embd = nn.Embedding(cfg.env.num_objects + 3, cfg.train.embd_dim)
        self.convs = nn.Sequential(
            nn.Conv2d(cfg.train.embd_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.gru = nn.GRU(self.N * self.N * 64, cfg.train.hidden_size, num_layers=cfg.train.gru_layers,
                          batch_first=True)
        self.action_head = nn.Linear(cfg.train.hidden_size, 4)

    def forward(self, x, hx=None):
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

        # gru
        features, hx = self.gru(x, hx)

        # action head
        action_logits = self.action_head(features)

        return features, action_logits, hx

    def act(self, state=None, action=None, reward=None, hx=None):
        state, action, reward = map(lambda x: x[:, None] if x is not None else None, (state, action, reward))
        features, action_logits, hx = self.forward(state, hx=hx)
        action, log_prob = self.sample_action(action_logits)

        return PolicyOutput(action=action.squeeze(-1),
                            log_prob=log_prob.squeeze(-1),
                            features=features,
                            hidden_state=hx)
