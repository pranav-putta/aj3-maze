import torch
from torch import nn

from aj3.configs import MazeArguments
from einops import rearrange

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MazeNet(nn.Module):
    def __init__(self, cfg: MazeArguments, critic=False):
        super().__init__()
        self.N = cfg.agent_visibility
        if cfg.agent_visibility == -1:
            self.N = cfg.size
        self.tok_embd = nn.Embedding(cfg.num_objects + 3, 16)
        self.conv1 = nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 4, kernel_size=3, stride=1, padding=1)
        self.gru = nn.GRU(self.N * self.N * 4, 16, num_layers=2, batch_first=True)
        self.action_head = nn.Linear(16, 4)
        self.cfg = cfg

        if critic:
            self.critic_head = nn.Linear(16, 1)

    def forward(self, x):
        b, t, n, _ = x.shape
        x = x.to(device)
        x = rearrange(x, 'b t x y -> b t (x y)')
        x = self.tok_embd(x)
        x = rearrange(x, 'b t (x y) d -> (b t) d x y', x=n, y=n)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.flatten(1)
        x = rearrange(x, '(b t) d -> b t d', b=b)
        logits, _ = self.gru(x)
        action_dist = self.action_head(logits)

        if getattr(self, 'critic_head', None) is not None:
            value = self.critic_head(logits).squeeze(-1)
            return action_dist, value
        else:
            return action_dist
