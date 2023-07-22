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
        self.last_exp = None
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

    def to_tensordict(self, batch_first=True):
        """ stacks all tensors in storage into a TensorDict """
        batch = TensorDict({k: torch.stack(v) for k, v in self._storage.items()
                            if all([type(x) == torch.Tensor for x in v])},
                           batch_size=[self.current_step, self.num_envs])

        if batch_first:
            batch = batch.transpose(0, 1)
        return batch

    def compute_stats(self, gamma):
        batch = self.to_tensordict()
        batch = batch.to('cpu')
        assert 'rewards' in batch.keys(), 'rewards must be in storage'
        assert 'prev_dones' in batch.keys(), 'done_mask must be in storage'

        returns = compute_returns(next_value=None,
                                  rewards_t=batch['rewards'],
                                  done_mask_t=batch['prev_dones'],
                                  use_gae=False,
                                  gamma=gamma)
        batch['returns'] = returns

        num_episodes = 0
        successes = 0
        returns = 0
        lengths = 0
        for env in range(self.num_envs):
            start_idxs = torch.where(batch['prev_dones'][env])[0].tolist()
            end_idxs = start_idxs[1:] + torch.tensor([self.current_step]).tolist()
            truncated = set(torch.where(batch['truncated'][env])[0].tolist())

            # different computation when there is only one episode in the rollout and it was unsuccessful
            if len(start_idxs) == 1 and (end_idxs[0] - 1) in truncated and not batch['success'][env, end_idxs[0] - 1]:
                lengths += (end_idxs[0] - start_idxs[0])
                returns += batch['returns'][env, start_idxs[0]].item()
                num_episodes += 1
                continue

            for i, (start, end) in enumerate(zip(start_idxs, end_idxs)):
                if i == len(start_idxs) - 1 and end - 1 in truncated:
                    continue
                lengths += end - start
                returns += batch['returns'][env, start].item()
                num_episodes += 1
                successes += batch['success'][env, end - 1].item()

        return RolloutStats(success_rate=successes / num_episodes,
                            num_episodes=num_episodes,
                            avg_episode_length=lengths / num_episodes,
                            avg_reward=returns / num_episodes)

    def save_episode_to_mp4(self, filename, env=None):
        if env is None:
            env = random.randint(0, self.num_envs - 1)

        frames = []
        done = False
        max_width = 500

        def grid_to_frame(f, s, v, su, a):
            img = maze_to_rgb(f)
            img_height, img_width = img.shape[:2]
            ratio = max_width // img_width
            img_height, img_width = img_height * ratio, img_width * ratio
            img = Image.fromarray(img).resize((img_height, img_width), resample=Image.NEAREST)

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

            # Overlay the agent view on the image
            if True:
                partial_view_rgb = maze_to_rgb(s)
                partial_view_size = max_width // 6

                partial_height, partial_width = partial_view_rgb.shape[:2]
                ratio = partial_view_size // partial_width
                partial_height, partial_width = ratio * partial_height, ratio * partial_width

                small_map_x = padded_image.width - (partial_width // 2) - padding
                small_map_y = padded_image.height - (partial_height // 2) - padding

                p_img = Image.fromarray(partial_view_rgb).resize((partial_height, partial_width),
                                                                 resample=Image.NEAREST)
                # Add a border around the partial view image
                border_color = (255, 255, 255)
                border_thickness = 5
                p_img = ImageOps.expand(p_img, border=border_thickness, fill=border_color)
                padded_image.paste(p_img, (small_map_x, small_map_y))

            return padded_image

        while not done:
            step = len(frames)
            full_view = self._storage['infos'][step]['full_view'][env]
            state = self._storage['states'][step][env]
            valid = self._storage['infos'][step]['valid'][env]
            succ = self._storage['success'][step][env]
            act = self._storage['actions'][step][env].item()
            done = self._storage['prev_dones'][step + 1][env] if step + 1 < len(self._storage['prev_dones']) else True

            frames.append(grid_to_frame(full_view, state, valid, succ, act))

        frames_to_mp4(frames, filename)

    @staticmethod
    def minibatch_generator(batch, num_minibatches=1):
        B, T = batch['states'].shape[:2]
        new_batch_size = B // num_minibatches
        for inds in torch.randperm(B).chunk(num_minibatches):
            inds = inds.tolist()
            yield TensorDict({k: v[inds] for k, v in batch.items()},
                             batch_size=[new_batch_size, T])
