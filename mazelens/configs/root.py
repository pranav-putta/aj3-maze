from dataclasses import dataclass
from typing import Any, Tuple

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class BaseTrainerConfig:
    lr: float = 1e-4
    max_grad_norm: float = 0.5
    epochs: int = 10


@dataclass
class BaseEnvConfig:
    _target_: str
    env_id: str


@dataclass
class MazeEnvConfig(BaseEnvConfig):
    size: Tuple[int, int] = (9, 9)
    image_only_obs: bool = False
    top_camera: bool = False
    global_observables: bool = False


@dataclass
class PPOConfig(BaseTrainerConfig):
    clip_epsilon: float = 0.2


@dataclass
class BCConfig(BaseTrainerConfig):
    pass


@dataclass
class Config:
    env: MazeEnvConfig = MISSING
    trainer: BaseTrainerConfig = MISSING
    num_environments: int = 1
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)
cs.store(group="trainer", name="base_ppo", node=PPOConfig)
cs.store(group="trainer", name="base_bc", node=BCConfig)
cs.store(group='env', name='base_maze_env', node=MazeEnvConfig)
