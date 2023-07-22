import os

from gym import Env

os.environ['MUJOCO_GL'] = 'glfw'

# from memory_maze import tasks, _make_gym_env


class DMMazeEnv(Env):

    def __init__(self, size, two_d, **kwargs):
        super().__init__()

        self.env = None
        self.size = size
        if size == (9, 9):
            self.env = tasks.memory_maze_9x9
        elif size == (11, 11):
            self.env = tasks.memory_maze_11x11
        elif size == (13, 13):
            self.env = tasks.memory_maze_13x13
        elif size == (15, 15):
            self.env = tasks.memory_maze_15x15
        else:
            raise ValueError('Invalid maze size')
        self.env = _make_gym_env(self.env,
                                 image_only_obs=not two_d,
                                 top_camera=two_d,
                                 global_observables=True)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        raise NotImplementedError('Need to implement either 2D or 3D DMMazeEnv')

    def reset(self, **kwargs):
        raise NotImplementedError('Need to implement either 2D or 3D DMMazeEnv')


class DMMaze2DEnv(DMMazeEnv):
    def __init__(self, size, **kwargs):
        super().__init__(size, **kwargs)

    def reset(self, **kwargs):
        return


class DMMaze3DEnv(DMMazeEnv):
    def __init__(self, size, **kwargs):
        super().__init__(size, **kwargs)
