from einops import rearrange
from torch import nn

from mazelens.agents import AgentInput
from mazelens.nets.base_net import Net


class SimpleNet(Net):
    def __init__(self, in_dim, embd_vocab_size, embd_dim, rnn_layers, hidden_dim, out_dim):
        super().__init__()
        self.tok_embd = nn.Embedding(embd_vocab_size, embd_dim)
        self.convs = nn.Sequential(
            nn.Conv2d(embd_dim, 16, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        )
        self.gru = nn.GRU(in_dim * in_dim * 64, hidden_dim, num_layers=rnn_layers,
                          batch_first=True)
        self.action_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, agent_input: AgentInput, generation_mode=False):
        hx = agent_input.prev.hiddens
        x = agent_input.states
        b, t, n, _ = x.shape

        # embed tokens
        x = x.long()
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
