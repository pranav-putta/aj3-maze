import abc
import numpy as np


# abstract class agent which takes in an observation and outputs an action
class Agent(abc.ABC):
    def __init__(self, env):
        self.env = env

    @abc.abstractmethod
    def act(self, observation: np.ndarray) -> int:
        pass
