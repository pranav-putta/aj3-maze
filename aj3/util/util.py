import heapq
import random
from functools import lru_cache

import cv2
import numpy as np
import torch
import pickle

from mazelab.color_style import object_colors

from aj3.util.configs import MazeArguments

import gym


def get_trainer(trainer_name):
    from aj3.agents import REINFORCEAgent, PPOAgent, BCAgent, AlgorithmicAgent
    if trainer_name == 'reinforce':
        return REINFORCEAgent
    elif trainer_name == 'ppo':
        return PPOAgent
    elif trainer_name == 'bc':
        return BCAgent
    elif trainer_name == 'algorithmic':
        return AlgorithmicAgent


def get_policy_net(net_name):
    from aj3.policy import SimplePolicyNet, MyopicPolicyNet, ImpalaPolicyNet, DecisionTransformerPolicyNet, \
        RTGDecisionTransformerPolicyNet, ShortestPathPolicy
    if net_name == 'simple':
        return SimplePolicyNet
    elif net_name == 'myopic':
        return MyopicPolicyNet
    elif net_name == 'impala':
        return ImpalaPolicyNet
    elif net_name == 'gpt':
        return DecisionTransformerPolicyNet
    elif net_name == 'dt-rtg':
        return RTGDecisionTransformerPolicyNet
    elif net_name == 'shortest-path':
        return ShortestPathPolicy


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


def maze_to_rgb(grid):
    rgb = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    for x in range(grid.shape[0]):
        for y in range(grid.shape[1]):
            rgb[x, y, :] = object_colors[grid[x, y].item()]

    return rgb


def frames_to_mp4(frames, filename):
    """Save a list of frames as an mp4 file."""
    videodims = frames[0].size
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(filename, fourcc, 3, videodims)
    # draw stuff that goes on every frame here
    for frame in frames:
        video.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    video.release()
