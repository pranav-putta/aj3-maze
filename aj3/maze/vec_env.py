import multiprocessing as mp
import sys
from copy import deepcopy
from enum import Enum

import torch
from gym.error import AlreadyPendingCallError, NoAsyncCallError
from gym.vector import AsyncVectorEnv
from gym.vector.async_vector_env import AsyncState
from gym.vector.utils import (concatenate, write_to_shared_memory)


class VectorEnvState(Enum):
    WAITING_INFO = 'info'


def _worker_shared_memory(index, env_fn, pipe, parent_pipe, shared_memory, error_queue):
    assert shared_memory is not None
    env = env_fn()
    observation_space = env.observation_space
    parent_pipe.close()
    try:
        while True:
            command, data = pipe.recv()
            if command == 'reset':
                observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send((None, True))
            elif command == 'step':
                observation, reward, done, info = env.step(data)
                if done:
                    observation = env.reset()
                write_to_shared_memory(index, observation, shared_memory,
                                       observation_space)
                pipe.send(((None, reward, done, info), True))
            elif command == 'seed':
                env.seed(data)
                pipe.send((None, True))
            elif command == 'close':
                pipe.send((None, True))
                break
            elif command == '_check_observation_space':
                pipe.send((data == observation_space, True))
            elif command == 'info':
                data = env.info()
                pipe.send((data, True))
            else:
                raise RuntimeError('Received unknown command `{0}`. Must '
                                   'be one of {`reset`, `step`, `seed`, `close`, '
                                   '`_check_observation_space`}.'.format(command))
    except (KeyboardInterrupt, Exception):
        error_queue.put((index,) + sys.exc_info()[:2])
        pipe.send((None, False))
    finally:
        env.close()


class VectorEnv(AsyncVectorEnv):

    def __init__(self, env):
        super().__init__(env, worker=_worker_shared_memory)

    def info(self):
        self.info_async()
        return self.info_wait()

    def info_async(self):
        self._assert_is_running()
        if self._state != AsyncState.DEFAULT:
            raise AlreadyPendingCallError('Calling `reset_async` while waiting '
                                          'for a pending call to `{0}` to complete'.format(
                self._state.value), self._state.value)

        for pipe in self.parent_pipes:
            pipe.send(('info', None))
        self._state = VectorEnvState.WAITING_INFO

    def info_wait(self, timeout=None):
        self._assert_is_running()
        if self._state != VectorEnvState.WAITING_INFO:
            raise NoAsyncCallError('Calling `info_wait` without any prior '
                                   'call to `info_async`.', VectorEnvState.WAITING_INFO.value)

        if not self._poll(timeout):
            self._state = AsyncState.DEFAULT
            raise mp.TimeoutError('The call to `reset_wait` has timed out after '
                                  '{0} second{1}.'.format(timeout, 's' if timeout > 1 else ''))

        results, successes = zip(*[pipe.recv() for pipe in self.parent_pipes])
        self._raise_if_errors(successes)
        self._state = AsyncState.DEFAULT

        best_actions = torch.tensor(list(results))
        return best_actions
