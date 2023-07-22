import logging
from omegaconf import OmegaConf

import wandb


class Logger:
    def log(self, data):
        pass


class WandBLogger(Logger):
    def __init__(self, project, name, group, tags, notes, should_log, cfg=None):
        self.name = name
        self.group = group
        self.tags = tags
        self.notes = notes
        self.should_log = should_log

        if self.name == 'None':
            self.name = None

        if self.should_log:
            wandb.init(project=project,
                       config=cfg,
                       name=self.name,
                       group=self.group,
                       tags=self.tags,
                       notes=self.notes)

    def log(self, data):
        if self.should_log:
            wandb.log(data)


class TextLogger(Logger):
    def __init__(self, **kwargs):
        self.logger = logging.getLogger(__name__)

    def log(self, data):
        self.logger.info(data)
