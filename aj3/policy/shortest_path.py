import torch
from mazelab import VonNeumannMotion

from aj3.agents.agent import PolicyOutput
from aj3.policy.policy import Policy
from aj3.util.configs import MazeArguments


class ShortestPathPolicy(Policy):
    def __init__(self, cfg: MazeArguments):
        super().__init__(cfg)
        self.motions = VonNeumannMotion()

    def act(self, state=None, action=None, reward=None, hx=None, distance_map=None, agent_position=None):
        actions = []
        for env in range(self.cfg.train.num_envs):
            dist_map = distance_map[env]
            agent_pos = agent_position[env]

            next_positions = [(agent_pos[0] + motion[0], agent_pos[1] + motion[1]) for motion in self.motions]
            min_action = min(range(len(next_positions)), key=lambda x: dist_map[next_positions[x]])
            actions.append(min_action)

        return PolicyOutput(action=torch.tensor(actions, dtype=torch.long))

    def initial_hidden_state(self):
        return None
