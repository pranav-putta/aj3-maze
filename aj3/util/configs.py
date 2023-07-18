from mltoolkit import GeneralArguments
from mltoolkit.argparser import argclass
from dataclasses import field


@argclass
class TrainingArguments:
    num_steps: int = field(default=1000)
    num_minibatches: int = field(default=4)
    epsilon: float = field(default=0.2)
    value_coef: float = field(default=0.5)
    entropy_coef: float = field(default=0.01)
    max_steps: int = field(default=500)
    num_envs: int = field(default=8)
    hidden_size: int = field(default=256)
    scale: float = field(default=0.5)
    lr: float = field(default=1e-4)
    ppo_epochs: int = field(default=4)
    gru_layers: int = field(default=1)
    gamma: float = field(default=0.99)
    embd_dim: int = field(default=32)
    max_grad_norm: float = field(default=0.5)
    bc_teacher_trainer: str = field(default='shortest')
    bc_teacher_net: str = field(default='dt-rtg')
    weight_decay: float = field(default=0.01)
    deterministic: bool = field(default=False)
    use_gae: bool = field(default=True)
    single_rollout: bool = field(default=False)


@argclass
class EnvironmentArguments:
    max_steps: int = field(default=500)
    reward_type: str = field(default='sparse')
    num_objects: int = field(default=1)
    agent_visibility: int = field(default=3)
    grid_size: int = field(default=5)
    difficulty: str = field(default='hard')
    static_env: bool = field(default=False)


@argclass
class MazeArguments(GeneralArguments):
    log_every: int = field(default=10)
    verbose: bool = field(default=True)
    trainer_name: str = field(default='ppo')
    net_name: str = field(default='simple')
    exp_dir: str = field(default='experiments')

    train: TrainingArguments
    env: EnvironmentArguments

    @property
    def state_dim(self):
        if self.agent_visibility == -1:
            return self.size
        else:
            return self.agent_visibility
