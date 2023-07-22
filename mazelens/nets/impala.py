import torch
from einops import rearrange
from torch import nn

from mazelens.nets.base_net import Net, StatefulNet
from mazelens.nets.modules.impala_blocks import ImpalaModel
from mazelens.util.structs import ExperienceDict


class ImpalaPolicyNet(Net):
    def __init__(self, in_dim, embd_vocab_size, embd_dim, hidden_dim, scale, out_dim, condition_on, rnn: StatefulNet):
        super().__init__()
        self.impala = ImpalaModel(embd_dim, in_dim, hidden_dim, scale)
        self.rnn = rnn
        self.action_head = nn.Linear(hidden_dim, out_dim)
        self.condition_on = condition_on
        self.hidden_dim = hidden_dim

        if 's' in self.condition_on:
            self.tok_embd = nn.Embedding(embd_vocab_size, embd_dim)
        if 'a' in self.condition_on:
            self.action_embd = nn.Embedding(out_dim, hidden_dim)
        if 'r' in self.condition_on:
            self.reward_embd = nn.Linear(1, hidden_dim)

    def forward(self, inp: ExperienceDict):
        s, a, r = inp.states, inp.actions, inp.rewards
        assert s is not None, 'states must be provided'
        b, t, p, q = s.shape

        # embed tokens
        s = s.long()
        s = rearrange(s, 'b t x y -> b t (x y)')
        s = self.tok_embd(s)
        s = rearrange(s, 'b t (x y) d -> (b t) d x y', x=p, y=q)
        s = self.impala(s)
        s = rearrange(s, '(b t) d -> b t d', b=b)

        tokens = {'s': s}
        if 'a' in self.condition_on and a is not None:
            tokens['a'] = self.action_embd(a.long())
        if 'r' in self.condition_on and r is not None:
            r = rearrange(r, 'b ... -> b ... 1')
            tokens['r'] = self.reward_embd(r.float())

        if t == 1:
            # in the case of a single step, we need to modify the order of the tokens to match the order of the
            # environment outputs.
            rotated_cond = self.condition_on
            if rotated_cond == 'sa':
                rotated_cond = 'as'
            elif rotated_cond == 'sar':
                rotated_cond = 'ars'
            elif rotated_cond == 'rsa':
                rotated_cond = 'rsa'
            keys, x = zip(*[(k, tokens[k]) for k in rotated_cond if k in tokens])
            x = torch.cat(x, dim=1)
        else:
            keys, x = zip(*[(k, tokens[k]) for k in self.condition_on if k in tokens])
            # interleave tokens into a sequence
            x = torch.stack(x, dim=2).view(b, t * len(x), self.hidden_dim)

        head_position = keys.index('s')
        x, hx = self.rnn(x, inp.prev_hiddens, inp.prev_dones)
        x = x[:, head_position::len(tokens), :]  # transformed state logits to pass to action head

        action_logits = self.action_head(x)
        return x, action_logits, hx
