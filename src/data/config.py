"""Data configuration."""
from dataclasses import dataclass


@dataclass
class DataConfig:
    task_type: str = "bigram"
    batch_size: int = 128
    eval_batch_size: int = 1024
    sequence_length: int = 64
    # Bigram-specific
    vocab_size: int = 64
    # LDS-specific
    dim_x: int = 64
    dim_y: int = 2
    lambda_val: float = 0.9
    structure: str = "dense"
    sigma: float = 1.0
    # Physics-specific
    physics_system: str = "harmonic"
    dt: float = 0.05
    obs_noise_std: float = 0.1
    x0_std: float = 1.0
    eval_sequence_length: int = 0  # 0 = same as sequence_length
    omega: float = 1.0
    zeta: float = 0.1
    n_oscillators: int = 3
    coupling_strength: float = 0.5
    gravity: float = 9.81
    observe_positions: bool = True
    observe_velocities: bool = False
