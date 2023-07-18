from gym import Env
from memory_maze import tasks, _make_gym_env
from torchrl.envs import TransformedEnv, Compose, ObservationNorm, DoubleToFloat, StepCounter


class MazeEnv(Env):

    def __init__(self, size, image_only_obs, top_camera, global_observables, **kwargs):
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
        self.env = _make_gym_env(self.env, image_only_obs=image_only_obs, top_camera=top_camera,
                                 global_observables=global_observables)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def step(self, action):
        return self.env.step(action)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class Maze2DEnv(MazeEnv):
    def __init__(self, size, **kwargs):
        super().__init__(size, **kwargs)

    @staticmethod
    def get_transform(env):
        return TransformedEnv(
            env,
            Compose(
                ObservationNorm(in_keys=['maze_layout']),
                DoubleToFloat(in_keys=['maze_layout']),
                StepCounter(),
            )
        )
