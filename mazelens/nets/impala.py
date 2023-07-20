import torch
from einops import rearrange
from torch import nn

from mazelens.agents import AgentInput
from mazelens.nets.base_net import Net
from mazelens.nets.modules.impala_blocks import ImpalaModel


class ImpalaPolicyNet(Net):
    def __init__(self, in_dim, embd_vocab_size, embd_dim, hidden_dim, scale, rnn_layers, out_dim):
        super().__init__()
        self.tok_embd = nn.Embedding(embd_vocab_size, embd_dim)
        self.impala = ImpalaModel(embd_dim, in_dim, hidden_dim, scale)
        self.gru = nn.GRU(hidden_dim, hidden_dim, num_layers=rnn_layers,
                          batch_first=True)

        self.action_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, agent_input: AgentInput):
        hx = agent_input.prev.hiddens
        x = agent_input.states
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
