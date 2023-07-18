import random
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image, ImageFont, ImageOps, ImageDraw
from einops import rearrange
from tensordict import TensorDict

from aj3.util.configs import MazeArguments
from aj3.util.util import maze_to_rgb, frames_to_mp4


@dataclass
class RolloutStats:
    success_rate: float
    num_episodes: int

    avg_episode_length: float
    avg_reward: float


class RolloutStorage:
    def __init__(self, cfg: MazeArguments):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.done_mask = []
        self.hidden_states = []
        self.successes = []
        self.valids = []

        self.cfg = cfg
        self.current_step = 0

    def insert(self, state, action, log_prob, reward, done, hidden_state, success, valid):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(list(reward))
        self.done_mask.append(list(done))
        self.hidden_states.append(hidden_state)
        self.successes.append(success)
        self.valids.append(valid)

        self.current_step += 1

    def compute_returns(self, next_value=None, rewards_t=None, done_mask_t=None, value_preds=None):
        if next_value is None:
            next_value = torch.zeros(self.cfg.train.num_envs, device=self.cfg.device)
        returns = [torch.zeros(self.cfg.train.num_envs, 1) for _ in range(self.current_step + 1)]
        returns[self.current_step] = next_value

        if self.cfg.train.use_gae and value_preds is not None:
            # if value_preds is None:
            #     raise ValueError("value_preds must be provided if using GAE")
            gae = 0.
            for step in reversed(range(self.current_step)):
                gamma = 0.99
                tau = 0.95
                delta = (rewards_t[:, step]
                         + gamma
                         * (value_preds[:, step + 1] if step + 1 < self.current_step else next_value)
                         * (~done_mask_t[:, step])
                         - value_preds[:, step])
                gae = delta + gamma * tau * (~done_mask_t[:, step]) * gae
                returns[step] = gae + value_preds[:, step]
        else:
            for step in reversed(range(self.current_step)):
                returns[step] = (rewards_t[:, step]
                                 + self.cfg.train.gamma
                                 * returns[step + 1]
                                 * (~done_mask_t[:, step]))
        return torch.stack(returns[:-1]).transpose(0, 1)

    def to_batch(self):
        batch = TensorDict({'states': torch.from_numpy(np.array(self.states[:-1])).long(),
                            'actions': torch.stack(self.actions),
                            'rewards': torch.tensor(self.rewards, dtype=torch.float),
                            'done_mask': torch.tensor(self.done_mask),
                            'successes': torch.tensor(self.successes)},
                           batch_size=[self.cfg.train.max_steps])
        if all([type(h) == torch.Tensor for h in self.hidden_states]):
            batch['hidden_states'] = torch.stack(self.hidden_states)
        if all([type(v) == torch.Tensor for v in self.log_probs]):
            batch['log_probs'] = torch.stack(self.log_probs)

        batch = batch.apply(lambda v: v.to(self.cfg.device))
        batch = batch.apply(lambda v: rearrange(v, 't b ... -> b t ...'))
        return batch

    def data_generator(self, batch):
        num_mini_batches = self.cfg.train.num_mini_batches

    def compute_stats(self, batch):
        returns = self.compute_returns(None, batch['rewards'], batch['done_mask'])

        num_episodes = 0
        avg_returns = 0
        avg_episode_lengths = 0

        for env in range(self.cfg.train.num_envs):
            done_idxs = torch.where(batch['done_mask'][env])[0]
            starts = torch.cat([torch.tensor([0], device=self.cfg.device), done_idxs[:-1] + 1])
            ends = done_idxs
            avg_episode_lengths += (ends - starts).sum().item()
            avg_returns += returns[env, starts].sum().item()
            num_episodes += len(starts)

        avg_episode_length = avg_episode_lengths / num_episodes
        avg_returns = avg_returns / num_episodes
        sr = (batch['successes'].sum() / num_episodes).item()

        return RolloutStats(success_rate=sr,
                            num_episodes=num_episodes,
                            avg_episode_length=avg_episode_length,
                            avg_reward=avg_returns)

    def save_episode_to_mp4(self, filename, env=None):
        if env is None:
            env = random.randint(0, self.cfg.train.num_envs - 1)

        frames = []
        done = False
        max_width = 500

        def grid_to_frame(s, v, su, a):
            img = maze_to_rgb(s)
            img_height, img_width = img.shape[:2]
            ratio = max_width // img_width
            img = Image.fromarray(img).resize(((ratio * img_width), (ratio * img_height)), resample=Image.NEAREST)

            # Add padding to the image
            font_size = 32
            font = ImageFont.truetype("arial.ttf", size=font_size)  # Replace "arial.ttf" with your desired font file
            padding = 50
            if v and not su:
                padding_color = (0, 0, 0)  # White color, you can change it to your desired color
            elif v and su:
                padding_color = (0, 255, 0)
            else:
                padding_color = (255, 0, 0)
            padded_image = ImageOps.expand(img, border=padding, fill=padding_color)
            draw = ImageDraw.Draw(padded_image)
            text = "Action: {}".format(a)  # Replace "action" with your desired action text
            text_width, text_height = draw.textsize(text, font=font)
            x = (padded_image.width - text_width) // 2
            y = padded_image.height - padding - text_height * 2

            # Overlay the text on the image
            draw.text((x, y), text, fill=(255, 255, 255), font=font)
            return padded_image

        frames.append(grid_to_frame(self.states[0][env], True, False, None))
        while not done:
            step = len(frames)
            state = self.states[step + 1][env]
            valid = self.valids[step][env]
            succ = self.successes[step][env]
            act = self.actions[step][env]
            done = self.done_mask[step][env]

            frames.append(grid_to_frame(state, valid, succ, act))

        frames_to_mp4(frames, filename)
