import abc
import pickle
import random
import time
from collections import namedtuple
from dataclasses import dataclass, field

import numpy as np
import torch
from gym import Env
from gym.spaces import Box
from gym.spaces import Discrete

from mazelens.configs import Config
from mazelens.envs.generator_kruskal import Kruskal
from mazelens.util import maze_to_rgb, dijkstra_distance
from mazelens.util.colors import object_colors

goal_object = 0


class Reward(abc.ABC):
    @abc.abstractmethod
    def __call__(self, agent_position, valid, success):
        pass


class SparseReward(Reward):
    def __call__(self, agent_position, valid, success):
        if success:
            return +1
        elif not valid:
            return -0.75
        else:
            return -0.5


class DistanceToGoalReward(Reward):

    def __init__(self, distance_map, agent_pos):
        self.distance_map = distance_map
        self.max_distance = torch.max(torch.masked_fill(self.distance_map, torch.isinf(self.distance_map), -1))
        self.last_dist = self.distance_map[agent_pos[0]][agent_pos[1]]

    def __call__(self, agent_position, valid, success):
        if success:
            return +1
        elif not valid:
            return -1
        else:
            cur_dist = self.distance_map[agent_position[0]][agent_position[1]]
            if cur_dist < self.last_dist:
                self.last_dist = cur_dist
                return -0.2
            else:
                self.last_dist = cur_dist
                return -0.5


@dataclass
class Object:
    r"""Defines an object with some of its properties.

    An object can be an obstacle, free space or food etc. It can also have properties like impassable, positions.

    """
    name: str
    value: int
    rgb: tuple
    impassable: bool
    positions: list = field(default_factory=list)


class Maze:
    cfg: Config
    grid: np.ndarray
    _original_maze: np.ndarray

    def __init__(self, static_env, seed, agent_visibility, num_objects, size):
        self.static_env = static_env
        self.seed = seed
        self.agent_visibility = agent_visibility
        self.num_objects = num_objects
        self.size = list(size)
        self.grid = None

        self.generate_maze()

    def generate_maze(self):
        if self.static_env:
            np.random.seed(self.seed)
            random.seed(self.seed)

        if self.grid is None or not self.static_env:
            # only generate if the grid is not already generated or if the environment is not static
            w, h = self.size
            grid = Kruskal(w // 2, h // 2)
            self.grid = grid.generate()
            # if agent visibility is -1, then the agent can see the entire maze
            # pad the maze with 1s to avoid needing to check for out of bounds
            if self.agent_visibility != -1:
                self.grid = np.pad(self.grid, self.agent_visibility, constant_values=1)

            self._original_maze = self.grid.copy()
        else:
            self.grid = self._original_maze.copy()

        objects = self.make_objects()
        self.objects = namedtuple('Objects', map(lambda x: x.name, objects), defaults=objects)()

    def make_objects(self):
        free = Object('free', 0, object_colors[0], False, np.stack(np.where(self.grid == 0), axis=1))
        obstacle = Object('obstacle', 1, object_colors[1], True, np.stack(np.where(self.grid == 1), axis=1))
        agent = Object('agent', 2, object_colors[2], False, [])

        color_vals = list(object_colors.values())
        items = [Object(f'obj{i}', i + 3, (color_vals[i + 3]), False, []) for i in
                 range(self.num_objects)]
        items[goal_object].impassable = False

        objs = [free, obstacle, agent, *items]
        for obj in objs:
            obj.positions = set(tuple(pos) for pos in obj.positions)

        return objs

    def update_grid(self, old_pos, new_pos):
        self.objects.agent.positions = [new_pos]
        for obj in self.objects:
            if len(obj.positions) == 1:
                pos = obj.positions[0]
                self.grid[pos[0], pos[1]] = obj.value
            elif old_pos is not None and old_pos in obj.positions:
                self.grid[old_pos[0], old_pos[1]] = obj.value
        self.grid[new_pos[0], new_pos[1]] = self.objects.agent.value

    @property
    def agent_view(self):
        radius = self.agent_visibility
        if radius == -1:
            return self.grid
        else:
            x, y = self.objects.agent.positions[0]
            agent_view = self.grid[x - radius:x + radius + 1, y - radius:y + radius + 1]
            return agent_view


class AJ3MazeEnv(Env):
    def __init__(self, static_env=None, static_episode=None, difficulty=None,
                 reward_type=None, seed=None, agent_visibility=None,
                 num_objects=None, size=None, max_steps=None, **kwargs):
        super().__init__()
        self.static_env = static_env
        self.static_episode = static_episode
        self.difficulty = difficulty
        self.reward_type = reward_type
        self.seed = seed
        self.agent_visibility = agent_visibility
        self.num_objects = num_objects
        self.size = size

        self.maze = Maze(self.static_env, self.seed, self.agent_visibility, self.num_objects, self.size)
        self.motions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        self.max_steps = max_steps
        self.steps = 0

        self.observation_space = Box(low=0, high=len(self.maze.objects), shape=self.maze.size, dtype=np.uint8)
        self.action_space = Discrete(len(self.motions))

    @property
    def agent(self):
        return self.maze.objects.agent

    @property
    def goal(self):
        return self.maze.objects[goal_object + 3]

    @property
    def best_action(self):
        dist_map = self.distance_map
        agent_pos = self.maze.objects.agent.positions[0]

        next_positions = [(agent_pos[0] + motion[0], agent_pos[1] + motion[1]) for motion in self.motions]
        best_action = min(range(len(next_positions)), key=lambda x: dist_map[next_positions[x]])
        return best_action

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = (current_position[0] + motion[0], current_position[1] + motion[1])

        valid = self._is_valid(new_position)
        success = self._is_goal(new_position)

        # if step was a valid move, update the grid
        if valid:
            self.maze.update_grid(current_position, new_position)
        reward = self.reward_fn(new_position, valid, success)

        done = success
        self.steps += 1
        if self.steps >= self.max_steps:
            done = True
        obs = self.maze.agent_view.astype(np.uint8)
        return obs, reward, success, done, {'valid': valid,
                                            'best_action': self.best_action}

    def reset(self, **kwargs):
        self.steps = 0
        self.maze.generate_maze()

        # 1. find all free positions and place objects in random spots
        indices = np.argwhere(self.maze.grid == 0)
        indices = indices.tolist()
        if self.static_episode:
            random.seed(self.seed)
        else:
            random.seed(time.time())
        random.shuffle(indices)

        for i, obj in enumerate(self.maze.objects):
            if 'obj' in obj.name:
                obj.positions = [tuple(indices[i])]

        self.distance_map = dijkstra_distance(pickle.dumps(self.maze.grid),
                                              pickle.dumps(self.maze.objects[goal_object + 3].positions))
        self.max_distance = torch.max(torch.masked_fill(self.distance_map, torch.isinf(self.distance_map), -1))

        # 3. place agent in a random spot based on difficulty
        if self.difficulty == 'easy':
            min_dist = 0
            max_dist = int(0.2 * self.max_distance)
        elif self.difficulty == 'medium':
            min_dist = int(0.4 * self.max_distance)
            max_dist = int(0.6 * self.max_distance)
        elif self.difficulty == 'hard':
            min_dist = int(0.6 * self.max_distance)
            max_dist = int(1.0 * self.max_distance)
        else:
            raise ValueError('Invalid difficulty')

        agent_pos = None
        for idx in indices:
            if min_dist < self.distance_map[idx[0], idx[1]] <= max_dist:
                agent_pos = [tuple(idx)]
                break

        assert agent_pos is not None, 'Could not find a valid agent position'
        self.maze.update_grid(None, agent_pos[0])

        # 4. set up the reward function
        if self.reward_type == 'sparse':
            self.reward_fn = SparseReward()
        elif self.reward_type == 'distance_to_goal':
            self.reward_fn = DistanceToGoalReward(self.distance_map, list(self.agent.positions)[0])
        else:
            raise ValueError('Invalid reward type')

        obs = self.maze.agent_view.astype(np.uint8)
        return obs, {'valid': True, 'success': False, 'best_action': self.best_action}

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
        r = self.agent_visibility
        return maze_to_rgb(self.maze.grid[r:-r, r:-r])
