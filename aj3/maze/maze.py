import pickle
import random

import numpy as np
import torch
from gym.spaces import Box
from gym.spaces import Discrete
from mazelab import BaseEnv
from mazelab import BaseMaze
from mazelab import DeepMindColor as color
from mazelab import Object
from mazelab import VonNeumannMotion

from aj3.util.configs import MazeArguments
from aj3.maze.generator import Kruskal
from aj3.util.rewards import SparseReward, DistanceToGoalReward
from aj3.util.colors import object_colors
from aj3.util.util import dijkstra_distance

goal_object = 0


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
        grid = Kruskal(self.cfg.env.grid_size // 2, self.cfg.env.grid_size // 2)
        self.grid = grid.generate()

    @property
    def size(self):
        return self.grid.shape

    def make_objects(self):
        free = Object('free', 0, color.free, False, np.stack(np.where(self.grid == 0), axis=1))
        obstacle = Object('obstacle', 1, color.obstacle, True, np.stack(np.where(self.grid == 1), axis=1))
        agent = Object('agent', 2, color.agent, False, [])

        color_vals = list(object_colors.values())
        items = [Object(f'obj{i}', i + 3, (color_vals[i + 3]), False, []) for i in
                 range(self.cfg.env.num_objects)]
        items[goal_object].impassable = False

        objs = [free, obstacle, agent, *items]
        for obj in objs:
            obj.positions = set(tuple(pos) for pos in obj.positions)

        return objs

    def to_value(self):
        return super().to_value()


class Env(BaseEnv):
    def __init__(self, cfg: MazeArguments):
        super().__init__()
        self.cfg = cfg
        self.maze = Maze(self.cfg)
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

    @property
    def best_action(self):
        dist_map = self.distance_map
        agent_pos = self.maze.objects.agent.positions[0]

        next_positions = [(agent_pos[0] + motion[0], agent_pos[1] + motion[1]) for motion in self.motions]
        best_action = min(range(len(next_positions)), key=lambda x: dist_map[next_positions[x]])
        return best_action

    @property
    def agent_viewj(self):
        radius = self.cfg.env.agent_visibility
        if radius == -1:
            return self.current_grid
        else:
            x, y = self.maze.objects.agent.positions[0]
            agent_view = self.current_grid[x - radius:x + radius + 1, y - radius:y + radius + 1]
            return agent_view

    def update_grid(self, agent_last_pos, agent_new_pos, valid):

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = (current_position[0] + motion[0], current_position[1] + motion[1])

        valid = self._is_valid(new_position)
        success = self._is_goal(new_position)

        # if step was a valid move, update the grid
        if valid:
            self.maze.objects.agent.positions = [new_position]
            for object in self.maze.objects:
                if current_position in object.positions:
                    self.current_grid[agent_last_pos[0], agent_last_pos[1]] = object.value
            self.current_grid[agent_new_pos[0], agent_new_pos[1]] = self.agent.value

        reward = self.reward_fn(new_position, valid, success)

        agent_view = self.get_agent_view()

        return agent_view, reward, success, {'valid': valid,
                                             'success': success,  # include success separately bc timeout
                                             'best_action': self.get_best_next_action()}

    def reset(self):
        # 1. find all free positions and place objects in random spots
        indices = np.argwhere(self.maze.grid == 0)
        indices = indices.tolist()
        if self.cfg.env.static_env:
            random.seed(self.cfg.seed)
        random.shuffle(indices)

        for i, obj in enumerate(self.maze.objects):
            if 'obj' in obj.name:
                obj.positions = [tuple(indices[i])]

        # 2. compute the distance map
        self.distance_map = dijkstra_distance(pickle.dumps(self.maze.grid), pickle.dumps(self.goal.positions))
        self.max_distance = torch.max(torch.masked_fill(self.distance_map, torch.isinf(self.distance_map), -1))

        # 3. place agent in a random spot based on difficulty
        if self.cfg.env.difficulty == 'easy':
            min_dist = 0
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
            if min_dist < self.distance_map[idx[0], idx[1]] <= max_dist:
                self.maze.objects.agent.positions = [tuple(idx)]
                break

        # 4. set up the reward function
        if self.cfg.env.reward_type == 'sparse':
            self.reward_fn = SparseReward()
        elif self.cfg.env.reward_type == 'distance_to_goal':
            self.reward_fn = DistanceToGoalReward(self.distance_map, list(self.agent.positions)[0])
        else:
            raise ValueError('Invalid reward type')

        # 5. create the grid
        self.current_grid = self.maze.to_value().copy()

        # 6. pad maze according to agent visibility
        if self.cfg.env.agent_visibility != -1:
            self.current_grid = np.pad(self.current_grid, self.cfg.env.agent_visibility, mode='constant',
                                       constant_values=1)

        return self.get_agent_view()

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
