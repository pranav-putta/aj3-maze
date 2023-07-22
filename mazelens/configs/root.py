import os.path
from dataclasses import dataclass, field
from typing import Optional

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf

from mazelens.configs.agent import PPOAgentConfig, NetAgentConfig, BCAgentConfig, BaseAgentConfig
from mazelens.configs.env import AJ3MazeEnvConfig, DMMaze2DEnvConfig
from mazelens.configs.net import LinearHeadConfig, ImpalaPolicyNetConfig, RNNStateEncoder, \
    TransformerStateEncoder
from mazelens.configs.trainer import BaseTrainerConfig, AlgorithmicDistillationTrainerConfig


@dataclass(kw_only=True)
class LoggerConfig:
    should_log: bool = True
    project: str = 'mazelens'
    name: Optional[str] = MISSING
    group: str = MISSING
    tags: Optional[list] = None
    notes: Optional[str] = None


@dataclass(kw_only=True)
class WBLoggerConfig(LoggerConfig):
    _target_: str = 'mazelens.util.logger.WandBLogger'


@dataclass(kw_only=True)
class TextLoggerConfig(LoggerConfig):
    _target_: str = 'mazelens.util.logger.TextLogger'


@dataclass(kw_only=True)
class Config:
    trainer: BaseTrainerConfig = MISSING
    logger: LoggerConfig = MISSING

    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

    exp_dir: str = 'experiments'
    exp_name: str = MISSING

    seed: int = 0


cs = ConfigStore.instance()
cs.store(name="base_config", node=Config)

cs.store(group='env', name='base_aj3_maze', node=AJ3MazeEnvConfig)
cs.store(group='env', name='base_dm_maze2d', node=DMMaze2DEnvConfig)

cs.store(group='agent', name='base_agent', node=BaseAgentConfig)
cs.store(group='agent', name='base_ppo_agent', node=PPOAgentConfig)
cs.store(group='agent', name='base_bc_agent', node=BCAgentConfig)
cs.store(group='agent', name='base_net_agent', node=NetAgentConfig)

cs.store(group='trainer', name='base_trainer', node=BaseTrainerConfig)
cs.store(group='trainer', name='base_ad_trainer', node=AlgorithmicDistillationTrainerConfig)

cs.store(group='net', name='base_impala_net', node=ImpalaPolicyNetConfig)
cs.store(group='net', name='base_linear_head', node=LinearHeadConfig)
cs.store(group='net', name='base_rnn_state_encoder', node=RNNStateEncoder)
cs.store(group='net', name='base_transformer_state_encoder', node=TransformerStateEncoder)

cs.store(group='logger', name='base_wb_logger', node=WBLoggerConfig)
cs.store(group='logger', name='base_text_logger', node=TextLoggerConfig)


# resolvers
def slice_path(path, idx):
    s = slice(*[int(x) if x else None for x in idx.split(':')])
    return '/'.join(os.path.split(path)[s])


OmegaConf.register_new_resolver('slice_path', slice_path)
OmegaConf.register_new_resolver('cond', lambda cond, t, f: t if cond else f)
OmegaConf.register_new_resolver('mul', lambda x, y: x * y)
OmegaConf.register_new_resolver('add', lambda x, y: x + y)
OmegaConf.register_new_resolver('eq', lambda x, y: x == y)
OmegaConf.register_new_resolver('obs_dim', lambda radius, size: size if radius == -1 else radius * 2 + 1)
