import torch
from torch.distributions import Categorical

from einops import rearrange
from torch import nn

from mazelens.agents import Agent
from mazelens.nets.base_net import Net
from mazelens.nets.modules.focal_loss import FocalLoss
from mazelens.nets.modules.gpt import GPT
from mazelens.nets.modules.impala_blocks import ImpalaModel


class DecisionTransformerNet(Net):
    def __init__(self, in_dim, embd_vocab_size, embd_dim, layers, attn_heads, hidden_dim, out_dim, scale):
        super().__init__()

        self.state_embd = nn.Embedding(embd_vocab_size, embd_dim)
        self.action_embd = nn.Embedding(4 + 1, hidden_dim)  # add another dimension for masking
        self.reward_embd = nn.Linear(1, hidden_dim)

        self.impala = ImpalaModel(embd_dim, in_dim, hidden_dim, scale)

        self.gpt = GPT(
            block_size=1024,
            n_layer=layers,
            n_head=attn_heads,
            n_embd=hidden_dim,
            obs_dim=hidden_dim,
            mode='arbitrary'
        )
        self.action_head = nn.Linear(hidden_dim, out_dim)
        self.loss = FocalLoss(gamma=2, reduction='mean', ignore_index=4)

    def transform_batch_to_input(self, batch):
        B, T = batch.shape
        states, episode_lengths = [], []
        rewards = []
        actions = []
        for env in range(B):
            done_idxs = torch.where(batch['dones'][env])[0]
            starts = torch.cat([torch.tensor([0], device=batch['states'].device), done_idxs[:-1] + 1])
            ends = done_idxs
            for start, end in zip(starts, ends):
                states.append(batch['states'][env, start:end + 1])
                actions.append(batch['actions'][env, start:end + 1])
                rewards.append(batch['rewards'][env, start:end + 1])
                episode_lengths.append(end - start + 1)
        states = torch.nn.utils.rnn.pad_sequence(states, batch_first=True)
        actions = torch.nn.utils.rnn.pad_sequence(actions, batch_first=True, padding_value=4)
        rewards = torch.nn.utils.rnn.pad_sequence(rewards, batch_first=True)

        return Agent.construct_policy_input(states=states, actions=actions, rewards=rewards)

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

    def forward(self, x: AgentInput, generation_mode=False):
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
