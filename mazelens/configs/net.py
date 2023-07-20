from dataclasses import dataclass, MISSING


@dataclass(kw_only=True)
class BaseNetConfig:
    """ Base class for network configurations. """
    _target_: str = MISSING


@dataclass(kw_only=True)
class SimpleNetConfig(BaseNetConfig):
    _target_: str = "mazelens.nets.simple.SimpleNet"

    in_dim: int = MISSING
    embd_vocab_size: int = MISSING
    embd_dim: int = MISSING
    rnn_layers: int = MISSING
    hidden_dim: int = MISSING
    out_dim: int = MISSING


@dataclass(kw_only=True)
class ImpalaPolicyNetConfig(BaseNetConfig):
    _target_: str = "mazelens.nets.impala.ImpalaPolicyNet"

    in_dim: int = MISSING
    embd_vocab_size: int = MISSING
    embd_dim: int = MISSING
    hidden_dim: int = MISSING
    scale: float = MISSING
    rnn_layers: int = MISSING
    out_dim: int = MISSING


@dataclass(kw_only=True)
class DecisionTransformerNetConfig(BaseNetConfig):
    _target_: str = "mazelens.nets.dt.DecisionTransformerNet"

    in_dim: int = MISSING
    embd_vocab_size: int = MISSING
    embd_dim: int = MISSING
    hidden_dim: int = MISSING
    scale: float = MISSING
    out_dim: int = MISSING
    layers: int = MISSING
    attn_heads: int = MISSING


@dataclass(kw_only=True)
class LinearHeadConfig(BaseNetConfig):
    _target_: str = "torch.nn.Linear"
    in_features: int = MISSING
    out_features: int = MISSING
