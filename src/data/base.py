"""Base dataset class."""
from abc import ABC, abstractmethod
from typing import Any
import jax


class BaseDataset(ABC):
    """Abstract base class for sequence datasets."""

    def __init__(self, name: str):
        self.name = name

    @abstractmethod
    def get_batch(self, rng: jax.Array, batch_size: int) -> dict:
        """Return a batch with keys: inputs, targets, mask."""
        pass

    @abstractmethod
    def get_eval_batch(self, rng: jax.Array, batch_size: int) -> dict:
        """Return an evaluation batch."""
        pass

    @staticmethod
    @abstractmethod
    def loss_fn(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
        """Compute loss. Used as static arg in train_step JIT."""
        pass

    @abstractmethod
    def compute_metrics(
        self, logits: jax.Array, targets: jax.Array, mask: jax.Array
    ) -> dict[str, float]:
        """Compute evaluation metrics, prefixed with self.name/."""
        pass

    def evaluate_rollouts(self, state: Any, generate_fn: Any, config: Any) -> dict:
        """Evaluate autoregressive rollouts. Override in subclasses."""
        return {}
