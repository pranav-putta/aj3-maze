from dataclasses import dataclass

from omegaconf import MISSING

import mazelens.agents.ppo
from mazelens.configs.net import BaseNetConfig


@dataclass(kw_only=True)
class BaseAgentConfig:
    """ Base class for all agent configs """
    _target_: str = MISSING
    _partial_: bool = True

    deterministic: bool = False


@dataclass(kw_only=True)
class NetAgentConfig(BaseAgentConfig):
    """ Agent config for neural network policies """
    policy: BaseNetConfig = MISSING
    lr: float = 1e-4


@dataclass(kw_only=True)
class PPOAgentConfig(NetAgentConfig):
    """ PPO agent specific config """
    _target_: str = "mazelens.agents.ppo.PPOAgent"
    critic: BaseNetConfig = MISSING
    epsilon: float = 0.2
    ppo_epochs: int = 4
    num_minibatches: int = 2
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    tau: float = 0.95
    use_gae: bool = True
