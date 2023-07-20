import random
from dataclasses import dataclass

import torch
from PIL import Image, ImageFont, ImageOps, ImageDraw
from tensordict import TensorDict

from mazelens.util.util import maze_to_rgb, frames_to_mp4, compute_returns


@dataclass
class RolloutStats:
    success_rate: float
    num_episodes: int

    avg_episode_length: float
    avg_reward: float


class RolloutStorage:
    def __init__(self, num_envs, *keys):
        self.num_envs = num_envs
        self._storage = {k: [].copy() for k in keys}
        self.current_step = 0

    def insert(self, **kwargs):
        if len(kwargs.keys()) != len(self._storage.keys()):
            raise ValueError("Number of keys does not match number of storage slots")
        for k, v in kwargs.items():
            self._storage[k].append(v)
        self.current_step += 1

    def add_field(self, key, value):
        assert type(value) == list, "field value must be a list"
        assert len(value) == self.current_step, "field value must be same length as storage"
        self._storage[key] = value

    def to_tensordict(self):
        """ stacks all tensors in storage into a TensorDict """
        return TensorDict({k: torch.stack(v) for k, v in self._storage.items()
                           if all([type(x) == torch.Tensor for x in v])},
                          batch_size=[self.current_step])

    def compute_stats(self, gamma):
        batch = self.to_tensordict()
        assert 'rewards' in batch.keys(), 'rewards must be in storage'
        assert 'dones' in batch.keys(), 'done_mask must be in storage'
        assert 'successes' in batch.keys(), 'successes must be in storage'

        returns = compute_returns(next_value=None,
                                  rewards_t=batch['rewards'],
                                  done_mask_t=batch['dones'],
                                  use_gae=False,
                                  gamma=gamma)

        num_episodes = 0
        avg_returns = 0
        avg_episode_lengths = 0

        for env in range(self.num_envs):
            done_idxs = torch.where(batch['dones'][:, env])[0]
            starts = torch.cat([torch.tensor([0]), done_idxs[:-1] + 1])
            ends = done_idxs
            avg_episode_lengths += (ends - starts).sum().item()
            avg_returns += returns[starts, env].sum().item()
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
            env = random.randint(0, self.num_envs - 1)

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
            font = ImageFont.truetype("misc/arial.ttf", size=font_size)
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

        while not done:
            step = len(frames)
            state = self._storage['states'][step][env]
            valid = self._storage['valids'][step][env] if step > 0 else True
            succ = self._storage['successes'][step][env] if step > 0 else False
            act = self._storage['actions'][step][env]
            done = self._storage['dones'][step][env] if step > 0 else False

            frames.append(grid_to_frame(state, valid, succ, act))

        frames_to_mp4(frames, filename)

    @staticmethod
    def minibatch_generator(batch, num_minibatches=1):
        T, B = batch['states'].shape[:2]
        new_batch_size = B // num_minibatches
        for inds in torch.randperm(B).chunk(num_minibatches):
            inds = inds.tolist()
            yield TensorDict({k: v[:, inds] for k, v in batch.items()},
                             batch_size=[T, new_batch_size])
