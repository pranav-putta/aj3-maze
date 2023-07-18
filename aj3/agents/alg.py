import torch

from aj3.agents.agent import Agent
from aj3.util.storage import RolloutStorage


class AlgorithmicAgent(Agent):
    def __init__(self, env, policy, cfg):
        super().__init__(env, policy, cfg)
        self.motions = env.motions

    def update_policy(self, rollouts):
        pass

    def parameters(self):
        return []

    def collect_episodes(self, envs):
        num_steps = self.cfg.train.max_steps
        rollouts = RolloutStorage(self.cfg)

        curr_state = envs.reset()
        hidden_state = self.policy.initial_hidden_state()
        for i in range(num_steps):
            curr_state = torch.tensor(curr_state, dtype=torch.long)
            acts = envs.info()
            next_states, rwds, dones, infos = envs.step(acts.cpu().numpy())

            success = [info['success'] for info in infos]
            valids = [info['valid'] for info in infos]

            rollouts.insert(curr_state, acts, None, rwds, dones, hidden_state, success, valids)

            curr_state = next_states
            if type(hidden_state) == torch.Tensor:
                hidden_state *= ~torch.tensor(dones, device=self.cfg.device).repeat(self.cfg.train.gru_layers,
                                                                                    self.cfg.train.hidden_size,
                                                                                    1).transpose(1, 2)

        # push the last state in the rollouts
        rollouts.states.append(torch.tensor(curr_state, dtype=torch.long))
        return rollouts
