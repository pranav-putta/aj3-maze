import heapq
import pickle
from functools import lru_cache

import gym
import numpy as np
import torch
from hydra.utils import get_class
from rich.console import Console
from rich.syntax import Syntax

from mazelens.configs.env import BaseEnvConfig
from mazelens.util.colors import object_colors
from functools import partial as f

import cv2


def print_yaml(yaml):
    console = Console()
    console.print(Syntax(yaml, "yaml"))


def maze_to_rgb(grid):
    rgb = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            rgb[x, y, :] = object_colors[grid[x, y].item()]

    return rgb


@lru_cache
def dijkstra_distance(pickled_maze, pickled_goal):
    goal = pickle.loads(pickled_goal)
    maze = pickle.loads(pickled_maze)
    rows, cols = maze.shape
    maze = torch.tensor(maze)
    distances = torch.empty((rows, cols)).fill_(float('inf'))
    distances[goal[0][0], goal[0][1]] = 0

    heap = [(0, goal[0])]
    visited = set()

    while heap:
        current_dist, current_pos = heapq.heappop(heap)
        if current_pos in visited:
            continue

        visited.add(current_pos)

        neighbors = []

        # Check neighboring positions
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x = current_pos[0] + dx
            new_y = current_pos[1] + dy

            if new_x >= 0 and new_x < rows and new_y >= 0 and new_y < cols:
                neighbors.append((new_x, new_y))

        for neighbor in neighbors:
            if maze[neighbor] == 0:
                neighbor_dist = current_dist + 1
                if neighbor_dist < distances[neighbor]:
                    distances[neighbor] = neighbor_dist
                    heapq.heappush(heap, (neighbor_dist, neighbor))

    return distances


def frames_to_mp4(frames, filename):
    """Save a list of frames as an mp4 file."""
    videodims = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, 3, videodims)
    # draw stuff that goes on every frame here
    for frame in frames:
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()


def compute_returns(next_value, rewards_t, done_mask_t, use_gae, gamma=None, tau=None, values_t=None):
    E, T = rewards_t.shape[:2]
    returns = [torch.zeros(E, ) for _ in range(T + 1)]
    next_value = next_value if next_value is not None else torch.zeros_like(rewards_t[:, 0])
    returns[T] = next_value

    if use_gae:
        if values_t is None:
            raise ValueError("value_preds must be provided if using GAE")
        gae = 0.
        for step in reversed(range(T)):
            not_done = ~done_mask_t[:, step + 1] if step + 1 < T else True
            delta = (rewards_t[:, step]
                     + gamma
                     * (values_t[:, step + 1] if step + 1 < T else next_value)
                     * not_done
                     - values_t[:, step])
            gae = delta + gamma * tau * not_done * gae
            returns[step] = gae + values_t[:, step]
    else:
        for step in reversed(range(T)):
            not_done = ~done_mask_t[:, step + 1] if step + 1 < T else True
            returns[step] = (rewards_t[:, step]
                             + gamma
                             * returns[step + 1]
                             * not_done)
    return torch.stack(returns[:-1]).transpose(0, 1)


def shift_tensor_sequence(x, fill_value, dim):
    slice_idx = [slice(None)] * dim + [slice(0, 1)]
    fill = torch.full_like(x[slice_idx], fill_value)
    return torch.cat([fill, x[:-1]], dim=dim)
