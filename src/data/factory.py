"""Dataset factory."""
import jax
from src.data.config import DataConfig
from src.data.base import BaseDataset


def create_dataset(config: DataConfig, rng: jax.Array) -> BaseDataset:
    """Create dataset from config."""
    if config.task_type == "bigram":
        from src.data.bigram import BigramDataset
        return BigramDataset(
            vocab_size=config.vocab_size,
            sequence_length=config.sequence_length,
            rng=rng,
        )
    elif config.task_type == "lds":
        from src.data.lds import LDSDataset
        return LDSDataset(
            dim_x=config.dim_x,
            dim_y=config.dim_y,
            lambda_val=config.lambda_val,
            structure=config.structure,
            sigma=config.sigma,
            sequence_length=config.sequence_length,
            rng=rng,
        )
    elif config.task_type == "physics":
        from src.data.physics import (
            HarmonicOscillatorDataset, DampedOscillatorDataset,
            CoupledOscillatorsDataset, ProjectileDataset,
        )
        cls_map = {
            "harmonic": HarmonicOscillatorDataset,
            "damped": DampedOscillatorDataset,
            "coupled": CoupledOscillatorsDataset,
            "projectile": ProjectileDataset,
        }
        if config.physics_system not in cls_map:
            raise ValueError(f"Unknown physics_system: {config.physics_system}")
        dataset = cls_map[config.physics_system](config=config, rng=rng)
        # Update config dims from physics (derived, not user-specified)
        config.dim_x = dataset.dim_x
        config.dim_y = dataset.dim_y
        return dataset
    else:
        raise ValueError(f"Unknown task_type: {config.task_type}")
