from dataclasses import dataclass, MISSING

from mazelens.configs.agent import BaseAgentConfig
from mazelens.configs.env import BaseEnvConfig


@dataclass(kw_only=True)
class BaseTrainerConfig:
    """ Base class for all trainer configs """
    _target_: str = "mazelens.trainers.Trainer"

    env: BaseEnvConfig = MISSING
    agent: BaseAgentConfig = MISSING

    lr: float = 1e-4
    max_grad_norm: float = 0.5
    epochs: int = 1000
    num_environments: int = 1
    num_rollout_steps: int = 100
    eval_frequency: int = 50
    gamma: float = 0.99
    tau: float = 0.95
    use_gae: bool = False
    log_videos: bool = False


@dataclass(kw_only=True)
class PPOTrainerConfig(BaseTrainerConfig):
    """ PPO trainer specific config """
    clip_epsilon: float = 0.2


@dataclass(kw_only=True)
class BCTrainerConfig(BaseTrainerConfig):
    """ BC trainer specific config """
    pass
