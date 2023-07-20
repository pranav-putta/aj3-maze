from dataclasses import dataclass

import torch
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING

from mazelens.configs.agent import PPOAgentConfig, NetAgentConfig, BCAgentConfig
from mazelens.configs.env import AJ3MazeEnvConfig, DMMaze2DEnvConfig
from mazelens.configs.net import SimpleNetConfig, LinearHeadConfig, ImpalaPolicyNetConfig, DecisionTransformerNetConfig
from mazelens.configs.trainer import BaseTrainerConfig, AlgorithmicDistillationTrainerConfig


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
cs.store(group='agent', name='base_bc_agent', node=BCAgentConfig)
cs.store(group='agent', name='base_net_agent', node=NetAgentConfig)

cs.store(group='trainer', name='base_trainer', node=BaseTrainerConfig)
cs.store(group='trainer', name='base_ad_trainer', node=AlgorithmicDistillationTrainerConfig)

cs.store(group='net', name='base_simple_net', node=SimpleNetConfig)
cs.store(group='net', name='base_impala_net', node=ImpalaPolicyNetConfig)
cs.store(group='net', name='base_decision_transformer_net', node=DecisionTransformerNetConfig)
cs.store(group='net', name='base_linear_head', node=LinearHeadConfig)
