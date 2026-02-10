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
    else:
        raise ValueError(f"Unknown task_type: {config.task_type}")
