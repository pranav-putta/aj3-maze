import torch
from torch.distributions import Categorical

from einops import rearrange
from torch import nn

class RTGDecisionTransformerPolicyNet(Policy):
    def __init__(self, cfg: MazeArguments):
        super().__init__(cfg)

        self.N = cfg.env.agent_visibility if cfg.env.agent_visibility != -1 else cfg.env.grid_size
        self.state_embd = nn.Embedding(cfg.env.num_objects + 3, cfg.train.embd_dim)
        self.action_embd = nn.Embedding(4 + 1, cfg.train.hidden_size)  # add another dimension for masking
        self.reward_embd = nn.Linear(1, cfg.train.hidden_size)

        self.impala = ImpalaModel(cfg.train.embd_dim, self.N, cfg.train.hidden_size, cfg.train.scale)

        self.gpt = GPT(
            block_size=1024,
            n_layer=2,
            n_head=4,
            n_embd=cfg.train.hidden_size,
            num_actions=4,
            obs_dim=cfg.train.hidden_size,
            mode='arbitrary'
        )
        self.action_head = nn.Linear(cfg.train.hidden_size, 4)
        self.loss = FocalLoss(gamma=2, reduction='mean', ignore_index=4)

    def batch_into_episodes(self, batch):
        states, episode_lengths = [], []
        rewards = []
        actions = []
        for env in range(self.cfg.train.num_envs):
            done_idxs = torch.where(batch['done_mask'][env])[0]
            starts = torch.cat([torch.tensor([0], device=self.cfg.device), done_idxs[:-1] + 1])
            ends = done_idxs
            for start, end in zip(starts, ends):
                states.append(batch['states'][env, start:end + 1])
                actions.append(batch['actions'][env, start:end + 1])
                rewards.append(batch['rewards'][env, start:end + 1])
                episode_lengths.append(end - start + 1)
        states = torch.nn.utils.rnn.pad_sequence(states, batch_first=True)
        actions = torch.nn.utils.rnn.pad_sequence(actions, batch_first=True, padding_value=4)
        rewards = torch.nn.utils.rnn.pad_sequence(rewards, batch_first=True)

        return states, actions, rewards

    def compute_features(self, states=None, actions=None, rewards=None,
                         input_order=('s', 'a', 'r'), hx=None,
                         return_pastkv=False, head_position=0):

        tokens = {}
        if states is not None:
            x = states
            b, t, n, _ = x.shape
            x = x.to(self.cfg.device)

            # embed tokens
            x = rearrange(x, 'b t x y -> b t (x y)')
            x = self.state_embd(x)
            x = rearrange(x, 'b t (x y) d -> (b t) d x y', x=n, y=n)

            # convolve
            x = self.impala(x)
            x = x.flatten(1)
            x = rearrange(x, '(b t) d -> b t d', b=b)

            tokens['s'] = x
        if actions is not None:
            actions = actions.to(self.cfg.device).long()
            actions = self.action_embd(actions)
            tokens['a'] = actions
        if rewards is not None:
            rewards = rewards.to(self.cfg.device).float()
            rewards = self.reward_embd(rewards.unsqueeze(-1))
            tokens['r'] = rewards

        gpt_input = [tokens[k] for k in input_order if k in tokens]
        features = self.gpt(tokens=gpt_input, past_kv=hx, return_past_kv=return_pastkv)
        features.logits = features.logits[head_position]
        return features

    def act(self, state=None, action=None, reward=None, hx=None):
        state, action, reward = map(lambda x: x[:, None] if x is not None else None, (state, action, reward))
        features = self.compute_features(states=state, actions=action,
                                         rewards=reward, input_order=('a', 'r', 's'),
                                         hx=hx, return_pastkv=True, head_position=-1)
        action_logits = self.action_head(features.logits)
        dist = Categorical(logits=action_logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        return PolicyOutput(action=action.squeeze(-1),
                            log_prob=log_prob.squeeze(-1),
                            features=features,
                            hidden_state=features.keys_values)

    def forward(self, batch, hx=None):
        states, actions, rewards = self.batch_into_episodes(batch)
        features = self.compute_features(states, actions, rewards, hx=hx)

        action_logits = self.action_head(features.logits)
        loss = self.loss(rearrange(action_logits, 'b t d -> (b t) d'), rearrange(actions, 'b t -> (b t)'))

        return features, action_logits, loss

    def initial_hidden_state(self):
        return self.gpt.empty_key_values(self.cfg.train.num_envs)
