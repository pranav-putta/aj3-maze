from dataclasses import dataclass, MISSING
from typing import Tuple


@dataclass
class BaseEnvConfig:
    """ Defines the base configuration for maze environments."""
    _target_: str = MISSING
    env_id: str = MISSING
    max_steps: int = MISSING


@dataclass
class MazeEnvConfig(BaseEnvConfig):
    size: Tuple[int, int] = MISSING


@dataclass
class AJ3MazeEnvConfig(BaseEnvConfig):
    """ Extends the base maze environment configuration with AJ3Maze specific parameters."""
    _target_: str = "mazelens.envs.AJ3MazeEnv"
    _partial_: bool = True
    env_id: str = "AJ3MazeEnv-v0"
    size: Tuple[int, int] = (9, 9)
    static_env: bool = True
    static_episode: bool = False
    agent_visibility: int = -1
    num_objects: int = 1
    difficulty: str = 'easy'
    reward_type: str = 'sparse'
    max_steps: int = 100


@dataclass
class DMMaze2DEnvConfig(MazeEnvConfig):
    """ Extends the base maze environment configuration with DMMaze specific parameters."""
    _target_: str = "mazelens.envs.DMMaze2DEnv"
    env_id: str = "DMMaze2D-v1"
    max_steps: int = 100
    size: Tuple[int, int] = (9, 9)
