import heapq
import random
from functools import lru_cache

import numpy as np
import torch
import pickle

from aj3.agents.ppo import PPOAgent

from aj3.agents.reinforce import REINFORCEAgent

from aj3.configs import MazeArguments
from aj3.nets.impala import ImpalaPolicyNet
from aj3.nets.myopic import MyopicPolicyNet
from aj3.nets.simple import SimplePolicyNet

from torch import nn
import gym


def get_trainer(cfg: MazeArguments):
    if cfg.trainer_name == 'reinforce':
        return REINFORCEAgent
    elif cfg.trainer_name == 'ppo':
        return PPOAgent


def get_policy_net(cfg: MazeArguments):
    if cfg.net_name == 'simple':
        return SimplePolicyNet
    elif cfg.net_name == 'myopic':
        return MyopicPolicyNet
    elif cfg.net_name == 'impala':
        return ImpalaPolicyNet


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def set_global_log_levels(level):
    gym.logger.set_level(level)


def adjust_lr(optimizer, init_lr, timesteps, max_timesteps):
    lr = init_lr * (1 - (timesteps / max_timesteps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def get_n_params(model):
    return str(np.round(np.array([p.numel() for p in model.parameters()]).sum() / 1e6, 3)) + ' M params'


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
