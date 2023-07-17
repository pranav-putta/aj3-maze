import pickle
import random

import numpy as np
import torch

from aj3.configs import MazeArguments
from aj3.maze.generator import DungeonRooms
from aj3.rewards import SparseReward, DistanceToGoalReward
from mazelab import BaseMaze
from mazelab import Object
from mazelab import DeepMindColor as color
from mazelab.color_style import object_colors
from mazelab.generators import double_t_maze

from mazelab import BaseEnv
from mazelab import VonNeumannMotion

import gym
from gym.spaces import Box
from gym.spaces import Discrete

from mltoolkit.argparser import parse_config

from aj3.util import dijkstra_distance

goal_object = 0


def hex_to_rgb(color_code):
    r, g, b = int(color_code[1:3], 16), int(color_code[3:5], 16), int(color_code[5:7], 16)
    return r, g, b


class Maze(BaseMaze):
    cfg: MazeArguments

    def __init__(self, config):
        self.cfg = config
        self.grid = None
        self.generate_maze()

        super().__init__()

    def generate_maze(self):
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        grid = DungeonRooms(self.cfg.env.grid_size // 2, self.cfg.env.grid_size // 2)
        self.grid = grid.generate()

    @property
    def size(self):
        return self.grid.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.grid == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.grid == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])

        color_vals = list(object_colors.values())
        items = [Object(f'obj{i}', i + 3, hex_to_rgb(color_vals[i + 3]), False, []) for i in
                 range(self.cfg.env.num_objects)]
        items[goal_object].impassable = False

        objs = [free, obstacle, agent, *items]
        for obj in objs:
            obj.positions = set(tuple(pos) for pos in obj.positions)

        return objs

    def center_grid_around_agent(self, grid, n):
        agent_coords = np.argwhere(grid == 2)  # Find the coordinates of the agent (assuming it is labeled as 2)

        if len(agent_coords) == 0:
            raise ValueError("Agent not found in the grid.")

        agent_row, agent_col = agent_coords[0]

        # Calculate padding size
        padding = n // 2

        # Create a new padded grid
        padded_grid = np.pad(grid, padding, mode='constant')

        # Calculate the center coordinates of the new grid
        center_row = agent_row + padding
        center_col = agent_col + padding

        # Extract the n x n grid centered around the agent
        centered_grid = padded_grid[center_row - padding: center_row + padding + 1,
                        center_col - padding: center_col + padding + 1]

        return centered_grid

    def to_value(self):
        if self.cfg.env.agent_visibility == -1:
            return super().to_value()
        return self.center_grid_around_agent(super().to_value(), self.cfg.env.agent_visibility)


class Env(BaseEnv):
    def __init__(self, cfg: MazeArguments):
        super().__init__()

        self.cfg = cfg
        self.maze = Maze(cfg)
        self.motions = VonNeumannMotion()
        self.current_grid = None
        self.distance_map = None

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

    @property
    def agent(self):
        return self.maze.objects.agent

    @property
    def goal(self):
        return self.maze.objects[goal_object + 3]

    def update_grid(self, agent_last_pos, agent_new_pos):
        for object in self.maze.objects:
            if agent_last_pos in object.positions:
                self.current_grid[agent_last_pos[0], agent_last_pos[1]] = object.value
        self.current_grid[agent_new_pos[0], agent_new_pos[1]] = self.agent.value

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = (current_position[0] + motion[0], current_position[1] + motion[1])
        valid = self._is_valid(new_position)
        success = self._is_goal(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        reward = self.reward(new_position, valid, success)
        self.update_grid(current_position, new_position)
        return self.current_grid, reward, success, {'valid': valid, 'success': self._is_goal(new_position)}

    def reset(self):
        indices = np.argwhere(self.maze.grid == 0)
        indices = indices.tolist()
        random.seed(self.cfg.seed)
        random.shuffle(indices)

        # randomly place objects in locations
        for i, obj in enumerate(self.maze.objects):
            if 'obj' in obj.name:
                obj.positions = [tuple(indices[i])]

        self.distance_map = dijkstra_distance(pickle.dumps(self.maze.grid), pickle.dumps(self.goal.positions))
        self.max_distance = torch.max(torch.masked_fill(self.distance_map, torch.isinf(self.distance_map), -1))

        # place agent based on difficulty
        if self.cfg.env.difficulty == 'easy':
            min_dist = 1
            max_dist = int(0.2 * self.max_distance)
        elif self.cfg.env.difficulty == 'medium':
            min_dist = int(0.4 * self.max_distance)
            max_dist = int(0.6 * self.max_distance)
        elif self.cfg.env.difficulty == 'hard':
            min_dist = int(0.6 * self.max_distance)
            max_dist = int(1.0 * self.max_distance)
        else:
            raise ValueError('Invalid difficulty')

        for idx in indices:
            if min_dist <= self.distance_map[idx[0], idx[1]] <= max_dist:
                self.maze.objects.agent.positions = [tuple(idx)]
                break

        if self.cfg.env.reward_type == 'sparse':
            self.reward = SparseReward()
        elif self.cfg.env.reward_type == 'distance_to_goal':
            self.reward = DistanceToGoalReward(self.distance_map, list(self.agent.positions)[0])
        else:
            raise ValueError('Invalid reward type')

        self.current_grid = self.maze.to_value().copy()
        return self.current_grid

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        for obj in self.maze.objects:
            if obj.impassable and tuple(position) in obj.positions:
                return False
        return nonnegative and within_edge

    def _is_goal(self, position):
        out = False
        for pos in self.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()
