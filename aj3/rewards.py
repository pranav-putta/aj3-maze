import abc
import heapq

import numpy as np
import pickle
import torch

from aj3.util import dijkstra_distance


class Reward(abc.ABC):
    @abc.abstractmethod
    def __call__(self, agent_position, valid, success):
        pass


class SparseReward(Reward):
    def __call__(self, agent_position, valid, success):
        if success:
            return +1
        elif not valid:
            return -1
        else:
            return -0.1


class DistanceToGoalReward(Reward):

    def __init__(self, distance_map, agent_pos):
        self.distance_map = distance_map
        self.max_distance = torch.max(torch.masked_fill(self.distance_map, torch.isinf(self.distance_map), -1))
        self.last_dist = self.distance_map[agent_pos[0]][agent_pos[1]]

    def __call__(self, agent_position, valid, success):
        if success:
            return +2
        elif not valid:
            return -1
        else:
            cur_dist = self.distance_map[agent_position[0]][agent_position[1]]
            if cur_dist < self.last_dist:
                return 0.1
            else:
                return -0.1
