from dataclasses import asdict
from dataclasses import fields

import gym
from gym.vector import AsyncVectorEnv

from mazelens.agents import Agent
from mazelens.util.storage import RolloutStorage
from mazelens.util.structs import ExperienceDict


class RolloutEnvWrapper(gym.Wrapper):
    env: AsyncVectorEnv

    def rollout(self, agent: Agent, num_steps) -> RolloutStorage:
        """ Standard implementation for rollout """
        states, infos = self.reset()
        exp = ExperienceDict(prev_dones=[True] * self.env.num_envs,
                             prev_hiddens=agent.initialize_hidden(self.env.num_envs),
                             states=states,
                             infos=infos,
                             actions=None, rewards=None, success=None, truncated=None)
        storage = RolloutStorage(self.env.num_envs, *[f.name for f in fields(exp)])
        # Experience Tuples: (prev_done, prev_hidden, state, info, action, reward, success, truncated)
        for step in range(num_steps):
            # before step transformation
            exp = agent.before_step(exp)

            # compute action and step environment
            actions, next_hidden = agent.act(exp)
            next_states, rwds, success, truncated, next_infos = self.step(actions.cpu())

            if step == num_steps - 1:
                truncated = [True] * self.env.num_envs
            next_dones = [d or t for d, t in zip(success, truncated)]

            # update experience tuple, and apply after step transformation
            exp.actions = actions
            exp.rewards = rwds
            exp.truncated = truncated
            exp.success = success
            exp = agent.after_step(exp)
            storage.insert(**asdict(exp))

            # update experience tuple
            # noinspection PyTypeChecker
            exp = ExperienceDict(prev_dones=next_dones,
                                 prev_hiddens=next_hidden,
                                 states=next_states,
                                 infos=next_infos,
                                 actions=actions,
                                 rewards=rwds,
                                 truncated=None, success=None)

        # insert the last incomplete experience tuple separately
        storage.last_exp = exp

        return storage
