"""Bigram (Markov chain) dataset for next-token prediction."""
import jax
import jax.numpy as jnp
import numpy as np
from src.data.base import BaseDataset
from src.data.metrics import compute_accuracy, compute_cross_entropy, compute_perplexity


class BigramDataset(BaseDataset):
    """Random bigram transition matrix, next-token prediction task."""

    def __init__(self, vocab_size: int, sequence_length: int, seed: int = 42):
        super().__init__(name="bigram")
        self.vocab_size = vocab_size
        self.sequence_length = sequence_length

        # Generate random transition matrix via Dirichlet
        rng = np.random.default_rng(seed)
        # Each row is a probability distribution over next tokens
        alpha = np.ones(vocab_size)
        self.transition_matrix = rng.dirichlet(alpha, size=vocab_size)  # (V, V)
        self.transition_matrix_jax = jnp.array(self.transition_matrix)

    def _sample_sequences(self, rng: jax.Array, batch_size: int) -> jax.Array:
        """Sample sequences from the Markov chain."""
        rng, start_rng = jax.random.split(rng)
        # Random starting tokens
        tokens = jax.random.randint(start_rng, (batch_size,), 0, self.vocab_size)
        sequence = [tokens]

        for _ in range(self.sequence_length - 1):
            rng, step_rng = jax.random.split(rng)
            # Get transition probs for current tokens
            probs = self.transition_matrix_jax[tokens]  # (B, V)
            tokens = jax.random.categorical(step_rng, jnp.log(probs + 1e-10), axis=-1)
            sequence.append(tokens)

        return jnp.stack(sequence, axis=1)  # (B, T)

    def get_batch(self, rng: jax.Array, batch_size: int) -> dict:
        """Return batch for training. inputs: tokens[:-1], targets: tokens[1:]."""
        sequences = self._sample_sequences(rng, batch_size)
        inputs = sequences[:, :-1]
        targets = sequences[:, 1:]
        mask = jnp.ones_like(targets, dtype=jnp.float32)
        return {"inputs": inputs, "targets": targets, "mask": mask}

    def get_eval_batch(self, rng: jax.Array, batch_size: int) -> dict:
        return self.get_batch(rng, batch_size)

    @staticmethod
    def loss_fn(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
        """Cross-entropy loss for next-token prediction."""
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        one_hot = jax.nn.one_hot(targets, logits.shape[-1])
        per_token = -jnp.sum(log_probs * one_hot, axis=-1)
        return jnp.sum(per_token * mask) / jnp.maximum(jnp.sum(mask), 1.0)

    def compute_metrics(
        self, logits: jax.Array, targets: jax.Array, mask: jax.Array
    ) -> dict[str, float]:
        return {
            f"{self.name}/accuracy": float(compute_accuracy(logits, targets, mask)),
            f"{self.name}/loss": float(compute_cross_entropy(logits, targets, mask)),
            f"{self.name}/perplexity": float(compute_perplexity(logits, targets, mask)),
        }

    def evaluate_rollouts(self, state, generate_fn, config) -> dict:
        """Generate sequences autoregressively and compare to true bigram statistics."""
        rng = jax.random.PRNGKey(42)
        num_rollouts = config.eval.num_rollouts
        rollout_steps = config.eval.rollout_steps

        # Generate prompt tokens (single starting token per sequence)
        rng, start_rng = jax.random.split(rng)
        prompt = jax.random.randint(start_rng, (num_rollouts, 1), 0, self.vocab_size)

        # Autoregressive generation
        generated = generate_fn(
            state, prompt,
            max_new_tokens=rollout_steps,
            temperature=1.0,
            rng_key=rng,
            mode="discrete",
        )  # (num_rollouts, 1 + rollout_steps)

        # Compute empirical transition counts from generated sequences
        generated_np = np.array(generated)
        empirical_counts = np.zeros((self.vocab_size, self.vocab_size))
        for i in range(generated_np.shape[1] - 1):
            src = generated_np[:, i]
            dst = generated_np[:, i + 1]
            for s, d in zip(src, dst):
                empirical_counts[s, d] += 1

        # Normalize to get empirical transition probs
        row_sums = empirical_counts.sum(axis=1, keepdims=True)
        empirical_probs = empirical_counts / np.maximum(row_sums, 1.0)

        # KL divergence: D_KL(true || empirical) for rows with enough data
        kl_divs = []
        for i in range(self.vocab_size):
            if row_sums[i, 0] >= 10:  # Only rows with enough samples
                p = self.transition_matrix[i]
                q = empirical_probs[i]
                # KL(p || q) with smoothing
                q_smooth = (q + 1e-8)
                q_smooth = q_smooth / q_smooth.sum()
                kl = np.sum(p * np.log(p / q_smooth + 1e-10))
                kl_divs.append(kl)

        mean_kl = float(np.mean(kl_divs)) if kl_divs else 0.0

        # Sequence accuracy: fraction of transitions matching argmax of true distribution
        true_argmax = np.argmax(self.transition_matrix, axis=1)
        correct = 0
        total = 0
        for i in range(generated_np.shape[1] - 1):
            src = generated_np[:, i]
            dst = generated_np[:, i + 1]
            correct += np.sum(dst == true_argmax[src])
            total += len(src)

        mode_accuracy = float(correct / max(total, 1))

        return {
            f"{self.name}/rollout_kl_divergence": mean_kl,
            f"{self.name}/rollout_mode_accuracy": mode_accuracy,
        }
