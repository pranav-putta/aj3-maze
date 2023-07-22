"""
The MIT License (MIT) Copyright (c) 2020 Andrej Karpathy

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""
from dataclasses import dataclass

from einops import rearrange

"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

import numpy as np


@dataclass
class GPTOutput:
    logits: torch.Tensor = None
    keys_values: torch.Tensor = None


class GELU(nn.Module):
    def forward(self, input):
        return F.gelu(input)


class GPTConfig:
    """ base GPT config, params common to all GPT versions """
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1

    def __init__(self, vocab_size, block_size, **kwargs):
        self.vocab_size = vocab_size
        self.block_size = block_size
        for k, v in kwargs.items():
            setattr(self, k, v)


class GPT1Config(GPTConfig):
    """ GPT-1 like network roughly 125M params """
    n_layer = 12
    n_head = 12
    n_embd = 768


class BasicConfig:
    n_embd = 512
    n_head = 8
    n_layer = 4
    block_size = 1024
    embd_pdrop = 0.1
    resid_pdrop = 0.1
    attn_pdrop = 0.1
    vocab_size = 4
    max_timestep = 256


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, block_size):
        super().__init__()
        assert n_embd % n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(n_embd, n_embd)
        self.query = nn.Linear(n_embd, n_embd)
        self.value = nn.Linear(n_embd, n_embd)
        # regularization
        self.attn_drop = nn.Dropout(attn_pdrop)
        self.resid_drop = nn.Dropout(resid_pdrop)
        # output projection
        self.proj = nn.Linear(n_embd, n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        # self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
        #                              .view(1, 1, config.block_size, config.block_size))
        self.register_buffer("mask", torch.tril(torch.ones(block_size + 1, block_size + 1))
                             .view(1, 1, block_size + 1, block_size + 1))
        self.n_head = n_head

    def forward(self, x, layer_past):
        B, T, C = x.size()

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)  # (B, nh, T, hs)

        if layer_past is not None:
            pk, pv = layer_past
            k = torch.cat((pk, k), dim=2)
            v = torch.cat((pv, v), dim=2)

        seq_len = k.shape[2]
        past_len = (k.shape[2] - q.shape[2])

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:, :, past_len:seq_len, :seq_len] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y, (k, v)


class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, attn_pdrop, resid_pdrop, block_size):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.attn = CausalSelfAttention(n_embd=n_embd,
                                        n_head=n_head,
                                        attn_pdrop=attn_pdrop,
                                        resid_pdrop=resid_pdrop,
                                        block_size=block_size)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(resid_pdrop),
        )

    def forward(self, x, layer_past=None):
        attn, (k, v) = self.attn(self.ln1(x), layer_past)
        x = x + attn
        x = x + self.mlp(self.ln2(x))
        return x, (k, v)


class GPT(nn.Module):
    """  the full GPT language model, with a context size of block_size """

    def __init__(self, block_size=1024, max_timestep=500,
                 n_layer=6,
                 n_head=4,
                 n_embd=512,
                 num_actions=4,
                 embd_pdrop=0.1,
                 resid_pdrop=0.1,
                 attn_pdrop=0.1,
                 obs_dim=768,
                 mode="obs"):
        super().__init__()

        self.pos_emb = nn.Parameter(torch.zeros(1, block_size + 1, n_embd))
        self.drop = nn.Dropout(embd_pdrop)
        self.n_embd = n_embd
        self.n_head = n_head
        self.mode = mode

        # transformer
        self.blocks = nn.Sequential(*[Block(n_embd=n_embd,
                                            n_head=n_head,
                                            attn_pdrop=attn_pdrop,
                                            resid_pdrop=resid_pdrop,
                                            block_size=block_size)
                                      for _ in range(n_layer)])
        # decoder head
        self.ln_f = nn.LayerNorm(n_embd)
        # self.head = nn.Linear(n_embd, num_actions)

        self.block_size = block_size
        self.apply(self._init_weights)

        # logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

        if "reward" in mode:
            self.ret_emb = nn.Sequential(nn.Linear(1, n_embd), nn.Tanh())

        self.obs_embedding = nn.Sequential(nn.Linear(obs_dim, n_embd), GELU())
        nn.init.normal_(self.obs_embedding[0].weight, mean=0.0, std=0.02)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, weight_decay):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        no_decay.add('pos_emb')
        no_decay.add('global_pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(
            param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params),)

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups

    def empty_key_values(self, batch_size):
        return [(torch.zeros(batch_size, self.n_head, 0, self.n_embd // self.n_head, device=self.pos_emb.device),
                 torch.zeros(batch_size, self.n_head, 0, self.n_embd // self.n_head, device=self.pos_emb.device)) for _
                in range(len(self.blocks))]

    def forward_with_obs(self, observations, past_kv=None, return_past_kv=False):
        B, T, _ = observations.shape
        if past_kv is None:
            past_kv = self.empty_key_values(B)

        tokens_e = self.obs_embedding(observations)
        past_length = past_kv[0][0].shape[-2]
        pos_e = self.pos_emb[:, past_length:tokens_e.shape[1] + past_length, :]

        new_key_values = []

        x = self.drop(tokens_e + pos_e)
        for i, (layer_past, block) in enumerate(zip(past_kv, self.blocks)):
            x, (k, v) = block(x, layer_past=layer_past)
            if return_past_kv:
                new_key_values.append((k, v))
        x = self.ln_f(x)
        return GPTOutput(logits=x, keys_values=new_key_values)

    def forward_arbitrary(self, tokens, past_kv=None, return_past_kv=False):
        N = len(tokens)
        B, T, D = tokens[0].shape

        if past_kv is None:
            past_kv = self.empty_key_values(B)

        tokens_e = torch.stack(tokens, dim=2).view(B, T * N, D)
        past_length = past_kv[0][0].shape[-2]
        pos_e = self.pos_emb[:, past_length:tokens_e.shape[1] + past_length, :]

        new_key_values = []

        x = self.drop(tokens_e + pos_e)
        for i, (layer_past, block) in enumerate(zip(past_kv, self.blocks)):
            x, (k, v) = block(x, layer_past=layer_past)
            if return_past_kv:
                new_key_values.append((k, v))
        x = self.ln_f(x)
        # x = x.view(B, T, N, D).permute(2, 0, 1, 3)
        return GPTOutput(logits=x, keys_values=new_key_values)

    def forward(self, **kwargs):
        if self.mode == 'obs':
            return self.forward_with_obs(**kwargs)
        elif self.mode == 'arbitrary':
            return self.forward_arbitrary(**kwargs)
        else:
            raise NotImplementedError('Only obs mode is supported for now')
