import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mazelens.configs.root import WBLoggerConfig
from mazelens.envs import *
from mazelens.util.util import *
from mazelens.trainers import *
import warnings
import wandb


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: Config) -> None:
    print_yaml(OmegaConf.to_yaml(cfg))
    logger = instantiate(cfg.logger, cfg=cfg)
    trainer = instantiate(cfg.trainer, device=cfg.device, seed=cfg.seed, exp_dir=cfg.exp_dir,
                          exp_name=cfg.exp_name, logger=logger)
    stats = trainer.train()
    return stats.success_rate


if __name__ == "__main__":
    run()
