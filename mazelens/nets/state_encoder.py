import torch
from torch.distributions import Categorical

from einops import rearrange
from torch import nn

from mazelens.nets.base_net import Net
from mazelens.nets.modules.focal_loss import FocalLoss
from mazelens.nets.modules.gpt import GPT
from mazelens.nets.modules.impala_blocks import ImpalaModel


class TransformerStateEncoder(Net):
    def __init__(self, hidden_dim, layers, attn_heads):
        super().__init__()

        self.gpt = GPT(
            block_size=4096,
            n_layer=layers,
            n_head=attn_heads,
            n_embd=hidden_dim,
            obs_dim=hidden_dim,
            mode='arbitrary'
        )
        self.action_head = nn.Linear(hidden_dim, hidden_dim)
        self.loss = FocalLoss(gamma=2, reduction='mean', ignore_index=4)

    def compute_features(self, states=None, actions=None, rewards=None,
                         input_order=('s', 'a', 'r'), hx=None,
                         return_pastkv=True):

        tokens = {}
        if states is not None:
            x = states
            b, t, n, _ = x.shape

            # embed tokens
            x = rearrange(x, 'b t x y -> b t (x y)')
            x = x.long()
            x = self.state_embd(x)
            x = rearrange(x, 'b t (x y) d -> (b t) d x y', x=n, y=n)

            # convolve
            x = self.impala(x)
            x = x.flatten(1)
            x = rearrange(x, '(b t) d -> b t d', b=b)

            tokens['s'] = x
        if actions is not None:
            actions = actions.long()
            actions = self.action_embd(actions)
            if len(actions.shape) == 2:
                actions = actions[:, None]
            tokens['a'] = actions
        if rewards is not None:
            rewards = rewards.float()
            rewards = self.reward_embd(rewards.unsqueeze(-1))
            if len(rewards.shape) == 2:
                rewards = rewards[:, None]
            tokens['r'] = rewards

        keys, gpt_input = zip(*[(k, tokens[k]) for k in input_order if k in tokens])
        if hx is None:
            hx = self.gpt.empty_key_values(gpt_input[0].shape[0])
        features = self.gpt(tokens=gpt_input, past_kv=hx, return_past_kv=return_pastkv)
        head_position = keys.index('s')
        features.logits = features.logits[head_position]
        return features

    def single_forward(self, x, hx, mask):
        not_done_mask = ~mask
        pk, pv = hx

        # mask out previous hidden states accordingly
        # this is to prevent the rnn from using the hidden states of the previous episode
        pk = pk * not_done_mask[None, :, None]
        pv = pv * not_done_mask[None, :, None]

        # run transformer
        features = self.gpt(tokens=x, past_kv=(pk, pv), return_past_kv=True)
        return features.logits, features.past_kv






    def forward(self, x, hx):
        if generation_mode:
            input_order = ('a', 'r', 's')
        else:
            input_order = ('s', 'a', 'r')
        features = self.compute_features(x.states, x.prev_state.actions,
                                         x.prev_state.rewards, hx=x.prev_state.hiddens,
                                         input_order=input_order)

        action_logits = self.action_head(features.logits)

        return features, action_logits, features.keys_values

    def initial_hidden_state(self, batch_size):
        return self.gpt.empty_key_values(batch_size)
