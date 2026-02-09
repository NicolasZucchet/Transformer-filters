"""Data configuration."""
from dataclasses import dataclass


@dataclass
class DataConfig:
    task_type: str = "bigram"
    batch_size: int = 128
    eval_batch_size: int = 1024
    sequence_length: int = 64
    seed: int = 42
    # Bigram-specific
    vocab_size: int = 64
    # LDS-specific
    dim_x: int = 64
    dim_y: int = 2
    lambda_val: float = 0.9
    structure: str = "dense"
    sigma: float = 1.0
