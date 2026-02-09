"""Input layers for different data modalities."""
from typing import Any
import jax.numpy as jnp
import flax.linen as nn


class EmbeddingInput(nn.Module):
    """Embedding layer for discrete token inputs."""
    vocab_size: int
    model_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.Embed(
            num_embeddings=self.vocab_size,
            features=self.model_dim,
            dtype=self.dtype,
        )(x)


class LinearInput(nn.Module):
    """Linear projection for continuous vector inputs."""
    model_dim: int
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        return nn.Dense(self.model_dim, dtype=self.dtype)(x)


class PatchingInput(nn.Module):
    """Linear projection with patching and causal zero-prepend for time series."""
    model_dim: int
    patch_size: int = 1
    dtype: Any = jnp.float32

    @nn.compact
    def __call__(self, x):
        b, t, d = x.shape
        p = self.patch_size
        if p > 1:
            x = x.reshape(b, t // p, p * d)
        zero = jnp.zeros((b, 1, x.shape[-1]), dtype=x.dtype)
        x = jnp.concatenate([zero, x], axis=1)
        return nn.Dense(self.model_dim, dtype=self.dtype)(x)
