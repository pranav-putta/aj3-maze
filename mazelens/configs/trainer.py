from dataclasses import dataclass, MISSING

from mazelens.configs.agent import BaseAgentConfig
from mazelens.configs.env import BaseEnvConfig


@dataclass(kw_only=True)
class BaseTrainerConfig:
    """ Base class for all trainer configs """
    _target_: str = "mazelens.trainers.Trainer"

    env: BaseEnvConfig = MISSING
    agent: BaseAgentConfig = MISSING

    epochs: int = MISSING
    num_environments: int = MISSING
    num_rollout_steps: int = MISSING
    eval_frequency: int = MISSING
    log_videos: bool = True


@dataclass(kw_only=True)
class AlgorithmicDistillationTrainerConfig(BaseTrainerConfig):
    """ BC trainer specific config """
    _target_: str = "mazelens.trainers.AlgorithmicDistillationTrainer"

    teacher_agent: BaseAgentConfig = MISSING
