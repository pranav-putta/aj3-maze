import torch
from einops import rearrange
from torch import nn

from mazelens.nets.base_net import Net, StatefulNet
from mazelens.nets.modules.impala_blocks import ImpalaModel
from mazelens.util.structs import ExperienceDict


class ImpalaPolicyNet(Net):
    def __init__(self, in_dim, embd_vocab_size, embd_dim, hidden_dim, scale, out_dim, rnn: StatefulNet):
        super().__init__()
        self.tok_embd = nn.Embedding(embd_vocab_size, embd_dim)
        self.impala = ImpalaModel(embd_dim, in_dim, hidden_dim, scale)
        self.rnn = rnn
        self.action_head = nn.Linear(hidden_dim, out_dim)

    def forward(self, inp: ExperienceDict):
        x, hx, done_mask = inp.states, inp.prev_hiddens, inp.prev_dones
        b, t, p, q = x.shape

        # embed tokens
        x = x.long()
        x = rearrange(x, 'b t x y -> b t (x y)')
        x = self.tok_embd(x)
        x = rearrange(x, 'b t (x y) d -> (b t) d x y', x=p, y=q)

        # impala + gru
        x = self.impala(x)
        x = rearrange(x, '(b t) d -> b t d', b=b)
        feats, hx = self.rnn(x, hx, done_mask)

        action_logits = self.action_head(feats)
        return feats, action_logits, hx
