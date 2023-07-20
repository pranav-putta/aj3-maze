import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf

from mazelens.envs import *
from mazelens.util.util import *
from mazelens.trainers import *
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(cfg: Config) -> None:
    print_yaml(OmegaConf.to_yaml(cfg))

    trainer = instantiate(cfg.trainer, cfg.device, cfg.seed, cfg.exp_dir)
    trainer.train()


if __name__ == "__main__":
    run()
