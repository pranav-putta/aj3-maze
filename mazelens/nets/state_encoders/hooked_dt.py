import torch
from einops import rearrange
from torch import nn
from torch.nn.utils.rnn import unpad_sequence
from transformer_lens import HookedTransformerKeyValueCache

from mazelens.nets.base_net import StatefulNet
from mazelens.nets.modules.focal_loss import FocalLoss
from mazelens.nets.modules.gpt import GPT
from mazelens.nets.modules.hooked_gpt import HookedGPT
from mazelens.nets.modules.hooked_gpt_config import HookedGPTConfig


class HookedTransformerStateEncoder(StatefulNet):
    def __init__(self, n_layers, d_model, d_head, n_heads, d_mlp, act_fn, n_ctx, mode):
        super().__init__()

        self.gpt = HookedGPT({
            'n_layers': n_layers,
            'd_model': d_model,
            'd_head': d_head,
            'n_heads': n_heads,
            'd_mlp': d_mlp,
            'act_fn': act_fn,
            'normalization_type': 'LNPre',
            'n_ctx': n_ctx
        })
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

        # todo; right now, we're assuming hidden state is always reset so we just don't consider it
        num_tokens_per_state = T // dT
        done_mask = [done_mask] + [torch.zeros_like(done_mask)] * (num_tokens_per_state - 1)
        done_mask = torch.stack(done_mask, dim=2).view(B, T)
        if self.should_rebatch_inputs:
            x, lengths = self.construct_padded_sequence(x, done_mask)

        # we're going to assume we start at a reset state, so past_kv is None
        x, hx = self.gpt(x, past_kv_cache=hx)
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
        return HookedTransformerKeyValueCache.init_cache(self.gpt.cfg, self.gpt.cfg.device, batch_size)
