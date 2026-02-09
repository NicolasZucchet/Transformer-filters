"""Top-level configuration and CLI argument generation from dataclasses."""
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import get_type_hints

from src.model.config import ModelConfig
from src.data.config import DataConfig


@dataclass
class TrainConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_steps: int = 100
    total_steps: int = 5000
    log_interval: int = 50
    grad_clip: float = 1.0


@dataclass
class EvalConfig:
    eval_interval: int = 500
    do_rollout: bool = False
    num_rollouts: int = 256
    rollout_steps: int = 64
    warmup_len: int = 64
    n_eval: int = 8192


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    wandb_project: str = "transformer-filters"
    wandb_mode: str = "online"
    seed: int = 42


# --- Auto CLI argument generation ---

def _add_dataclass_args(parser: argparse.ArgumentParser, dc_class, prefix: str = ""):
    """Recursively add dataclass fields as CLI arguments."""
    hints = get_type_hints(dc_class)
    for f in dataclasses.fields(dc_class):
        name = f"{prefix}{f.name}" if prefix else f.name
        ftype = hints[f.name]

        # If the field is itself a dataclass, recurse
        if dataclasses.is_dataclass(ftype):
            _add_dataclass_args(parser, ftype, prefix=f"{name}.")
            continue

        # Determine argparse type and kwargs
        kwargs = {"dest": name, "default": None}

        if ftype is bool:
            kwargs["type"] = lambda x: x.lower() in ("true", "1", "yes")
            kwargs["metavar"] = "BOOL"
        elif ftype is int:
            kwargs["type"] = int
        elif ftype is float:
            kwargs["type"] = float
        elif ftype is str:
            kwargs["type"] = str
        else:
            kwargs["type"] = str

        parser.add_argument(f"--{name}", **kwargs)


def _apply_overrides(dc_instance, overrides: dict, prefix: str = ""):
    """Recursively apply CLI overrides to a dataclass instance."""
    for f in dataclasses.fields(dc_instance):
        name = f"{prefix}{f.name}" if prefix else f.name
        value = getattr(dc_instance, f.name)

        if dataclasses.is_dataclass(value):
            _apply_overrides(value, overrides, prefix=f"{name}.")
        elif name in overrides and overrides[name] is not None:
            setattr(dc_instance, f.name, overrides[name])


def parse_config(dc_class, description: str = ""):
    """Parse CLI args and return a populated config dataclass."""
    parser = argparse.ArgumentParser(description=description)
    _add_dataclass_args(parser, dc_class)
    args = parser.parse_args()
    overrides = vars(args)

    config = dc_class()
    _apply_overrides(config, overrides)
    return config
