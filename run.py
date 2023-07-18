import os

import gym

from mazelens.configs.root import MazeEnvConfig

os.environ['MUJOCO_GL'] = 'glfw'

import hydra
from omegaconf import OmegaConf

from mazelens.configs import Config
from mazelens.util.util import *
from functools import partial as f
from hydra.utils import get_class, instantiate


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: Config) -> None:
    print_yaml(OmegaConf.to_yaml(cfg))

    env = instantiate(cfg.env)



if __name__ == "__main__":
    run()
