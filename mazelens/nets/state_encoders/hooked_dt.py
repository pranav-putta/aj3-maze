import torch
from einops import rearrange
from torch.nn.utils.rnn import unpad_sequence

from mazelens.nets.base_net import StatefulNet
from mazelens.nets.modules.focal_loss import FocalLoss
from mazelens.nets.modules.hooked_gpt import HookedGPT
from mazelens.nets.modules.hooked_gpt_components import HookedTransformerKeyValueCache, \
    HookedTransformerKeyValueCacheEntry


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
        self.mode = mode

    @property
    def should_rebatch_inputs(self):
        if self.mode == 'continuous':
            return False
        elif self.mode == 'rebatch':
            return True
        else:
            raise Exception(f'invalid mode {self.mode}')

    def single_forward(self, x, hx, done_mask):
        B, _, D = x.shape

        if type(hx) == HookedTransformerKeyValueCache:
            hx = hx.separate_batch(B)

        xs, hxs = [], []
        for b in range(B):
            if done_mask[b]:
                hx[b] = self.initialize_hidden(1)
            x_b = self.gpt(x[b:b + 1], past_kv_cache=hx[b])
            xs.append(x_b)

        x = torch.cat(xs, dim=0)
        hx = [hx_i.detach_tensors() for hx_i in hx]
        return x, hx

    def test_multistep_past_kv_cache(self, x, cache):
        tmp_xs = []
        tmp_caches = []
        for i in range(2):
            tmp_xs.append(x.clone())
            tmp_caches.append(
                HookedTransformerKeyValueCache(entries=[HookedTransformerKeyValueCacheEntry(e.past_keys.clone(),
                                                                                            e.past_values.clone())
                                                        for e in cache.entries]))

        # try one large step
        out1 = self.gpt(tmp_xs[0], past_kv_cache=tmp_caches[0])

        # try in steps
        out2 = []
        for i in range(tmp_xs[1].shape[1]):
            out2.append(self.gpt(tmp_xs[1][:, i:i + 1], past_kv_cache=tmp_caches[1]))
        out2 = torch.cat(out2, dim=1)
        assert torch.allclose(out1, out2, rtol=1e-3)

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
        x = self.gpt(x, past_kv_cache=None)

        if self.should_rebatch_inputs:
            x = unpad_sequence(x, lengths, batch_first=True)
            x = torch.cat(x, dim=0)
            x = rearrange(x, '(b t) ... -> b t ...', b=B)

        return x, None

    def stack_hiddens_into_tensor(self, hiddens):
        return torch.stack([torch.stack(kv) for kv in hiddens])

    def unstack_hiddens_from_tensor(self, hiddens):
        L, K, *_ = hiddens.shape
        return [[hiddens[l, k] for k in range(K)] for l in range(L)]

    def initialize_hidden(self, batch_size):
        return HookedTransformerKeyValueCache.init_cache(self.gpt.cfg, self.gpt.cfg.device, batch_size)
