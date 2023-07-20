from dataclasses import asdict
from typing import Tuple, Any

import gym
from gym import Env
import abc

from gym.vector import AsyncVectorEnv

from mazelens.agents import Agent, AgentInput
from mazelens.util import compute_returns
from mazelens.util.storage import RolloutStorage


class RolloutEnvWrapper(gym.Wrapper):
    env: AsyncVectorEnv

    def rollout(self, agent: Agent, num_steps) -> RolloutStorage:
        """ Standard implementation for rollout """
        states, infos = self.reset()
        prev_agent_output = agent.initial_agent_output()

        storage = None

        for step in range(num_steps):
            # noinspection PyTypeChecker
            x = agent.transform_input(AgentInput(states=states,
                                                 infos=infos,
                                                 prev=prev_agent_output))
            agent_output = agent.act(x)
            next_states, rewards, successes, dones, next_infos = self.step(list(agent_output.actions.cpu()))

            output = agent.transform_output(states=states, rewards=rewards, infos=infos,
                                            successes=successes, dones=dones, agent_output=agent_output)
            if storage is None:
                storage = RolloutStorage(self.env.num_envs, *output.keys())

            storage.insert(**output)

            states = next_states
            infos = next_infos
            prev_agent_output = agent_output
            prev_agent_output.rewards = rewards

        storage.insert_last(states=states, infos=infos)

        return storage
