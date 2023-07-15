import abc
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PolicyOutput:
    action: int = field(default=None)
    log_prob: float = field(default=None)
    value: float = field(default=None)
    hidden_state: np.ndarray = field(default=None)


# abstract class agent which takes in an observation and outputs an action
class Agent(abc.ABC):
    def __init__(self, env):
        self.env = env

    @abc.abstractmethod
    def act(self, observation: np.ndarray) -> PolicyOutput:
        pass
