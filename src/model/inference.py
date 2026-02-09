"""Autoregressive generation with KV cache (prefill + decode)."""
from functools import partial
import jax
import jax.numpy as jnp


@partial(jax.jit, static_argnums=(3, 4, 5))
def _prefill_step_discrete(params, x, rng, max_len, apply_fn, temperature):
    """Prefill for discrete token generation."""
    from flax.linen import make_causal_mask
    mask = make_causal_mask(x)
    logits, cache = apply_fn(
        {"params": params}, x, mask=mask,
        deterministic=True, init_cache=True, max_seq_len=max_len,
    )
    next_logits = logits[:, -1, :]
    if temperature == 0.0:
        next_token = jnp.argmax(next_logits, axis=-1)
        new_rng = rng
    else:
        new_rng, key = jax.random.split(rng)
        next_token = jax.random.categorical(key, next_logits / temperature, axis=-1)
    return next_token, cache, new_rng


@partial(jax.jit, static_argnums=(5, 6))
def _decode_step_discrete(params, x, cache, rng, step_idx, apply_fn, temperature):
    """Decode one token for discrete generation."""
    logits, new_cache = apply_fn(
        {"params": params}, x, mask=None,
        deterministic=True, cache=cache, decode_step=step_idx,
    )
    next_logits = logits[:, -1, :]
    if temperature == 0.0:
        next_token = jnp.argmax(next_logits, axis=-1)
        new_rng = rng
    else:
        new_rng, key = jax.random.split(rng)
        next_token = jax.random.categorical(key, next_logits / temperature, axis=-1)
    return next_token, new_cache, new_rng


@partial(jax.jit, static_argnums=(3, 4))
def _prefill_step_continuous(params, x, rng, max_len, apply_fn):
    """Prefill for continuous vector generation."""
    logits, cache = apply_fn(
        {"params": params}, x,
        deterministic=True, init_cache=True, max_seq_len=max_len,
    )
    next_state = logits[:, -1:, :]  # (B, 1, D)
    return next_state, cache


@partial(jax.jit, static_argnums=(4,))
def _decode_step_continuous(params, x, cache, step_idx, apply_fn):
    """Decode one step for continuous generation."""
    logits, new_cache = apply_fn(
        {"params": params}, x,
        deterministic=True, cache=cache, decode_step=step_idx,
    )
    next_state = logits[:, -1:, :]  # (B, 1, D)
    return next_state, new_cache


def generate(
    state, prompt, max_new_tokens, temperature=1.0, rng_key=None, mode="discrete",
):
    """Autoregressive generation with KV cache.

    Args:
        state: TrainState with params and apply_fn
        prompt: (B, L) for discrete or (B, L, D) for continuous
        max_new_tokens: Number of new tokens/steps to generate
        temperature: Sampling temperature (discrete only, 0.0 for greedy)
        rng_key: JAX PRNGKey for sampling
        mode: "discrete" or "continuous"

    Returns:
        Full sequence including prompt: (B, L+max_new_tokens) or (B, L+max_new_tokens, D)
    """
    prompt = jnp.array(prompt)
    if rng_key is None:
        rng_key = jax.random.PRNGKey(0)

    if max_new_tokens == 0:
        return prompt

    if mode == "discrete":
        return _generate_discrete(state, prompt, max_new_tokens, temperature, rng_key)
    elif mode == "continuous":
        return _generate_continuous(state, prompt, max_new_tokens, rng_key)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def _generate_discrete(state, prompt, max_new_tokens, temperature, rng_key):
    """Discrete token generation with KV cache."""
    B, L = prompt.shape
    total_len = L + max_new_tokens

    next_token, cache, rng_key = _prefill_step_discrete(
        state.params, prompt, rng_key, total_len, state.apply_fn, temperature,
    )
    curr_seq = jnp.concatenate([prompt, next_token[:, None]], axis=1)

    for i in range(max_new_tokens - 1):
        x = next_token[:, None]
        next_token, cache, rng_key = _decode_step_discrete(
            state.params, x, cache, rng_key, L + i, state.apply_fn, temperature,
        )
        curr_seq = jnp.concatenate([curr_seq, next_token[:, None]], axis=1)

    return curr_seq


def _generate_continuous(state, prompt, max_new_tokens, rng_key):
    """Continuous vector generation with KV cache."""
    B, L, D = prompt.shape
    total_len = L + max_new_tokens

    next_state, cache = _prefill_step_continuous(
        state.params, prompt, rng_key, total_len, state.apply_fn,
    )
    curr_seq = jnp.concatenate([prompt, next_state], axis=1)

    for i in range(max_new_tokens - 1):
        x = next_state
        next_state, cache = _decode_step_continuous(
            state.params, x, cache, L + i, state.apply_fn,
        )
        curr_seq = jnp.concatenate([curr_seq, next_state], axis=1)

    return curr_seq
