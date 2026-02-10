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
        return _generate_continuous(state.params, prompt, max_new_tokens, state.apply_fn)
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


@partial(jax.jit, static_argnums=(2, 3))
def _generate_continuous(params, prompt, max_new_tokens, apply_fn):
    """Continuous vector generation with KV cache, fused via lax.scan."""
    B, L, D = prompt.shape
    total_len = L + max_new_tokens

    # Prefill
    logits, cache = apply_fn(
        {"params": params}, prompt,
        deterministic=True, init_cache=True, max_seq_len=total_len,
    )
    first_pred = logits[:, -1:, :]  # (B, 1, D)

    if max_new_tokens == 1:
        return jnp.concatenate([prompt, first_pred], axis=1)

    # Decode via lax.scan (fused into single XLA computation)
    def decode_step(carry, _):
        x, cache, step_idx = carry
        logits, new_cache = apply_fn(
            {"params": params}, x,
            deterministic=True, cache=cache, decode_step=step_idx,
        )
        next_state = logits[:, -1:, :]
        return (next_state, new_cache, step_idx + 1), next_state[:, 0, :]

    init_carry = (first_pred, cache, L)
    _, generated = jax.lax.scan(
        decode_step, init_carry, None, length=max_new_tokens - 1,
    )
    # generated: (max_new_tokens - 1, B, D)
    generated = jnp.moveaxis(generated, 0, 1)  # (B, max_new_tokens - 1, D)

    all_preds = jnp.concatenate([first_pred, generated], axis=1)
    return jnp.concatenate([prompt, all_preds], axis=1)
