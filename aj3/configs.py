from mltoolkit import GeneralArguments
from mltoolkit.argparser import argclass
from dataclasses import field


@argclass
class TrainingArguments:
    num_episodes: int = field(default=1000)
    epsilon: float = field(default=0.2)
    value_coef: float = field(default=0.5)
    entropy_coef: float = field(default=0.01)
    max_steps: int = field(default=500)
    num_envs: int = field(default=8)


@argclass
class MazeArguments(GeneralArguments):
    log_every: int = field(default=10)
    num_objects: int = field(default=1)
    agent_visibility: int = field(default=3)
    verbose: bool = field(default=True)
    size: int = field(default=5)
    max_env_steps: int = field(default=100)

    train: TrainingArguments
