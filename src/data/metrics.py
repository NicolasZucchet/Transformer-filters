"""Evaluation metrics."""
import jax
import jax.numpy as jnp


@jax.jit
def compute_accuracy(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    """Masked accuracy for classification."""
    preds = jnp.argmax(logits, axis=-1)
    correct = (preds == targets) * mask
    return jnp.sum(correct) / jnp.maximum(jnp.sum(mask), 1.0)


@jax.jit
def compute_cross_entropy(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    """Masked cross-entropy loss."""
    # logits: (B, T, V), targets: (B, T)
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    B, T = targets.shape
    one_hot = jax.nn.one_hot(targets, logits.shape[-1])
    per_token = -jnp.sum(log_probs * one_hot, axis=-1)  # (B, T)
    return jnp.sum(per_token * mask) / jnp.maximum(jnp.sum(mask), 1.0)


@jax.jit
def compute_perplexity(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    """Masked perplexity."""
    ce = compute_cross_entropy(logits, targets, mask)
    return jnp.exp(ce)


@jax.jit
def compute_mse(predictions: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    """Masked MSE for regression. mask: (B, T), predictions/targets: (B, T, D)."""
    sq_err = jnp.sum((predictions - targets) ** 2, axis=-1)  # (B, T)
    return jnp.sum(sq_err * mask) / jnp.maximum(jnp.sum(mask), 1.0)


@jax.jit
def compute_per_step_mse(predictions: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
    """Per-timestep MSE. Returns (T,) array."""
    sq_err = jnp.sum((predictions - targets) ** 2, axis=-1)  # (B, T)
    per_step = jnp.sum(sq_err * mask, axis=0) / jnp.maximum(jnp.sum(mask, axis=0), 1.0)
    return per_step
