import contextlib
from pprint import pprint

import gym
import torch
from gym.wrappers import TimeLimit
from mltoolkit import parse_args
from tqdm import tqdm

from aj3.agents.shortest_path import ShortestPathAgent
from aj3.util.configs import MazeArguments
from aj3.maze.maze import Env
from aj3.maze.vec_env import VectorEnv
from aj3.util.storage import RolloutStorage
from aj3.util.util import get_trainer, get_policy_net, setup_seed


def setup_gym(cfg: MazeArguments):
    env_id = "Maze-v1"
    gym.register(id=env_id, entry_point=Env, max_episode_steps=cfg.env.max_steps, kwargs={'cfg': cfg})
    return env_id


def setup(cfg):
    setup_seed(cfg.seed)
    env_id = setup_gym(cfg)
    env = gym.make(env_id)

    network = get_policy_net(cfg)
    trainer = get_trainer(cfg)
    teacher = ShortestPathAgent(env, None, cfg)

    agent = trainer(env, network(cfg), cfg)
    agent = agent.to(cfg.device)

    print('Running training on device: {}'.format(cfg.device))
    print(f'Number of agent parameters {sum(p.numel() for p in agent.parameters())}')

    envs = VectorEnv(
        [lambda: TimeLimit(Env(cfg), max_episode_steps=cfg.env.max_steps) for _ in range(cfg.train.num_envs)])
    return agent, teacher, cfg, envs


def rollout_mode(cfg):
    if cfg.trainer_name == "ppo":
        return torch.no_grad()
    else:
        return contextlib.nullcontext()


def collect_teacher_episodes(envs, agent, cfg: MazeArguments):
    num_steps = cfg.train.max_steps
    rollouts = RolloutStorage(cfg)

    curr_state = envs.reset()
    hidden_state = torch.zeros((cfg.train.gru_layers, cfg.train.num_envs, cfg.train.hidden_size), dtype=torch.float,
                               device=cfg.device)
    for i in range(num_steps):
        info = envs.info()
        inputs = list(zip(*info))
        acts = torch.tensor([agent.act(x) for x in inputs])

        next_states, rwd, dones, info = envs.step(acts)
        success = [x['success'] for x in info]
        valids = [x['valid'] for x in info]
        rollouts.insert(curr_state, acts, torch.tensor([0]), rwd, dones, hidden_state, success, valids)

        curr_state = next_states

    # push the last state in the rollouts
    rollouts.states.append(torch.tensor(curr_state, dtype=torch.long))
    return rollouts


def train(cfg):
    agent, teacher, cfg, envs = setup(cfg)
    base_env = gym.make('Maze-v1', cfg=cfg)

    # Training loop
    pbar = tqdm(enumerate(range(cfg.train.num_steps)), total=cfg.train.num_steps)
    with rollout_mode(cfg):
        rollout = collect_teacher_episodes(envs, teacher, cfg)

    for i, episode in pbar:
        agent.update_policy(rollout)

        if (i + 1) % cfg.log_every == 0:
            eval_rollout = agent.collect_episodes(envs)
            eval_stats = eval_rollout.compute_stats(eval_rollout.to_batch())
            eval_rollout.save_episode_to_mp4(f'videos/teacher_{i}.mp4')
            pprint(eval_stats)


def main():
    cfg: MazeArguments = parse_args(MazeArguments)
    train(cfg)


if __name__ == '__main__':
    main()
