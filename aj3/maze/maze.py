import random

import numpy as np

from aj3.configs import MazeArguments
from aj3.maze.generator import DungeonRooms
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
        grid = DungeonRooms(self.cfg.size // 2, self.cfg.size // 2)
        self.grid = grid.generate()

    @property
    def size(self):
        return self.grid.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.grid == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.grid == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])

        color_vals = list(object_colors.values())
        objects = [Object(f'obj{i}', i + 3, hex_to_rgb(color_vals[i + 3]), False, []) for i in
                   range(self.cfg.num_objects)]
        objects[goal_object].impassable = False

        return free, obstacle, agent, *objects

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
        if self.cfg.agent_visibility == -1:
            return super().to_value()
        return self.center_grid_around_agent(super().to_value(), self.cfg.agent_visibility)


class Env(BaseEnv):
    def __init__(self):
        super().__init__()

        maze_args: MazeArguments = MazeArguments(**parse_config('configs/maze.yaml'))
        self.cfg = maze_args

        self.maze = Maze(maze_args)
        self.motions = VonNeumannMotion()

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

    @property
    def agent(self):
        return self.maze.objects.agent

    @property
    def goal(self):
        return self.maze.objects[goal_object + 3]

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [current_position[0] + motion[0], current_position[1] + motion[1]]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]

        if self._is_goal(new_position):
            reward = +1
            done = True
        elif not valid:
            reward = -1
            done = False
        else:
            reward = -0.1
            done = False
        return self.maze.to_value(), reward, done, {}

    def reset(self):
        indices = np.argwhere(self.maze.grid == 0)
        indices = indices.tolist()
        random.seed(self.cfg.seed)
        random.shuffle(indices)

        self.agent.positions = [indices.pop()]

        # randomly place objects in locations
        for i, obj in enumerate(self.maze.objects):
            if 'obj' in obj.name:
                obj.positions = [indices[i]]

        return self.maze.to_value()

    def _is_valid(self, position):
        nonnegative = position[0] >= 0 and position[1] >= 0
        within_edge = position[0] < self.maze.size[0] and position[1] < self.maze.size[1]
        passable = not self.maze.to_impassable()[position[0]][position[1]]
        return nonnegative and within_edge and passable

    def _is_goal(self, position):
        out = False
        for pos in self.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()
