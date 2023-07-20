from dataclasses import dataclass
from typing import Any

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, DictConfig

from mazelens.configs.env import AJ3MazeEnvConfig, DMMaze2DEnvConfig, BaseEnvConfig
from mazelens.configs.agent import PPOAgentConfig, NetAgentConfig, BaseAgentConfig
from mazelens.configs.net import SimpleNetConfig, LinearHeadConfig
from mazelens.configs.trainer import PPOTrainerConfig, BCTrainerConfig, BaseTrainerConfig


@dataclass(kw_only=True)
class Config:
    trainer: BaseTrainerConfig = MISSING

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    exp_dir: str = 'experiments/default'
    seed: int = 0


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

cs.store(group='env', name='base_aj3_maze', node=AJ3MazeEnvConfig)
cs.store(group='env', name='base_dm_maze2d', node=DMMaze2DEnvConfig)

cs.store(group='agent', name='base_ppo_agent', node=PPOAgentConfig)
cs.store(group='agent', name='base_net_agent', node=NetAgentConfig)

cs.store(group='trainer', name='base_trainer', node=BaseTrainerConfig)
cs.store(group='trainer', name='base_ppo_trainer', node=PPOTrainerConfig)
cs.store(group='trainer', name='base_bc_trainer', node=BCTrainerConfig)

cs.store(group='net', name='base_simple_net', node=SimpleNetConfig)
cs.store(group='net', name='base_linear_head', node=LinearHeadConfig)
