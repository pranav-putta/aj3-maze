from mltoolkit import GeneralArguments
from mltoolkit.argparser import argclass
from dataclasses import field


@argclass
class TrainingArguments:
    num_episodes: int = field(default=1000)


@argclass
class MazeArguments(GeneralArguments):
    num_objects: int = field(default=1)
    agent_visibility: int = field(default=3)
    verbose: bool = field(default=True)
    size: int = field(default=5)

    train: TrainingArguments
