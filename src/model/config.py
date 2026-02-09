"""Model configuration."""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    architecture: str = "transformer"
    model_dim: int = 256
    num_heads: int = 4
    num_layers: int = 4
    dropout_rate: float = 0.1
    mlp_coefficient: int = 4
    activation: str = "gelu"
    positional_encoding: str = "rope"  # "rope" or "learned"
    max_seq_len: int = 512
    dtype: str = "float32"
    patch_size: int = 1
