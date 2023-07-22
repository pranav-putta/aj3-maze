import torch
from einops import rearrange
from torch import nn
from torch.nn.utils.rnn import unpad_sequence

from mazelens.nets.base_net import StatefulNet
from mazelens.nets.modules.focal_loss import FocalLoss
from mazelens.nets.modules.gpt import GPT


class TransformerStateEncoder(StatefulNet):
    def __init__(self, hidden_dim, layers, attn_heads, mode):
        super().__init__()

        self.gpt = GPT(
            block_size=4096,
            n_layer=layers,
            n_head=attn_heads,
            n_embd=hidden_dim,
        )
        self.loss = FocalLoss(gamma=2, reduction='mean', ignore_index=4)
        self.mode = mode

    @property
    def should_rebatch_inputs(self):
        if self.mode == 'continuous':
            return False
        elif self.mode == 'rebatch':
            return True

    def single_forward(self, x, hx, done_mask):
        B, _, D = x.shape
        empty_pk = self.initialize_hidden(B)
        L = len(hx)

        # to deal with varying lengths
        hx = [[[empty_pk[l][k][b][None, :] if done_mask[b] else hx[l][k][b]
                for k in range(2)] for l in range(L)]
              for b in range(B)]

        # run transformer
        xs, hxs = [], []
        for b in range(B):
            x_b, hx_b = self.gpt(tokens=[x[b][None, :]], past_kv=hx[b], return_past_kv=True)
            xs.append(x_b)
            hxs.append(hx_b)

        x = torch.cat(xs, dim=0)
        hx = [[[hxs[b][l][k].detach() for b in range(B)] for k in range(2)] for l in range(L)]
        return x, hx

    def seq_forward(self, x: torch.Tensor, hx, done_mask):
        B, T, *_ = x.shape
        B, dT = done_mask.shape

        # reshape hidden states; originally (num_blocks, 2, B, attn_dim, T, H)
        # todo; right now, we're assuming hidden state is always reset
        hx = self.stack_hiddens_into_tensor(self.initialize_hidden(B))
        num_blocks, _, B, attn_dim, hT, H = hx.shape
        hx = torch.zeros(num_blocks, 2, B, attn_dim, T, H, device=x.device)

        num_tokens_per_state = T // dT
        done_mask = [done_mask] + [torch.zeros_like(done_mask)] * (num_tokens_per_state - 1)
        done_mask = torch.stack(done_mask, dim=2).view(B, T)
        if self.should_rebatch_inputs:
            hx = rearrange(hx, 'L K B A T H -> B T L K A H')
            x, hx, lengths = self.construct_padded_sequence(x, done_mask, hx=hx)
            hx = rearrange(hx, 'B L K A H -> L K B A 1 H')

        # we're going to assume we start at a reset state, so past_kv is None
        x, hx = self.gpt(tokens=[x], past_kv=None, return_past_kv=True)
        hx = self.stack_hiddens_into_tensor(hx)

        if self.should_rebatch_inputs:
            x = unpad_sequence(x, lengths, batch_first=True)
            x = torch.cat(x, dim=0)
            x = rearrange(x, '(b t) ... -> b t ...', b=B)

        return x, hx

    def stack_hiddens_into_tensor(self, hiddens):
        return torch.stack([torch.stack(kv) for kv in hiddens])

    def unstack_hiddens_from_tensor(self, hiddens):
        L, K, *_ = hiddens.shape
        return [[hiddens[l, k] for k in range(K)] for l in range(L)]

    def initialize_hidden(self, batch_size):
        return self.gpt.empty_key_values(batch_size)
