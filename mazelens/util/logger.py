import logging

import torch
from omegaconf import OmegaConf

import wandb


class Logger:
    def log(self, data):
        pass


class WandBLogger(Logger):
    def __init__(self, project, name, group, tags, notes, cfg):
        self.project = project
        self.name = name
        self.group = group
        self.tags = tags
        self.notes = notes
        self.cfg = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

        wandb.init(project=self.project,
                   config=self.cfg,
                   name=self.name,
                   group=self.group,
                   tags=self.tags,
                   notes=self.notes)

    def log(self, data):
        wandb.log(data)


class TextLogger(Logger):
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)

    def log(self, data):
        self.logger.info(data)
