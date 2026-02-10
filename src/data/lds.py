"""LDS dataset with rich system generation and Kalman filter baseline."""
from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from src.data.base import BaseDataset
from src.data.metrics import compute_mse
from src.kalman import kalman_filter, kalman_filter_final_state


def generate_system_parameters(key, dim_x, dim_y, lambda_val, structure='dense'):
    """Generate system parameters A and C.

    Args:
        structure: 'dense', 'diagonal', or 'scalar'.
            - 'dense': Random real matrix with complex conjugate eigenvalues (oscillatory).
            - 'diagonal': Real diagonal matrix with real eigenvalues (exponential).
            - 'scalar': Real scalar matrix A = s * I (exponential).
    """
    k1, k2, k3, k4 = jax.random.split(key, 4)

    if structure == 'scalar':
        sign = jax.random.choice(k1, jnp.array([-1.0, 1.0]))
        mag = jax.random.uniform(k2, minval=lambda_val / 2, maxval=lambda_val)
        s = sign * mag
        A = s * jnp.eye(dim_x)

    elif structure == 'diagonal':
        signs = jax.random.choice(k1, jnp.array([-1.0, 1.0]), shape=(dim_x,))
        mags = jax.random.uniform(k2, shape=(dim_x,), minval=lambda_val / 2, maxval=lambda_val)
        A = jnp.diag(signs * mags)

    elif structure == 'dense':
        n_pairs = dim_x // 2
        radii = jax.random.uniform(k1, (n_pairs,), minval=lambda_val / 2, maxval=lambda_val)
        angles = jax.random.uniform(k2, (n_pairs,), minval=0, maxval=jnp.pi)

        blocks = []
        for r, th in zip(radii, angles):
            block = jnp.array([[r * jnp.cos(th), -r * jnp.sin(th)],
                               [r * jnp.sin(th), r * jnp.cos(th)]])
            blocks.append(block)

        if dim_x % 2 != 0:
            r = jax.random.uniform(k3, minval=lambda_val / 2, maxval=lambda_val)
            s = jax.random.choice(k4, jnp.array([-1.0, 1.0]))
            blocks.append(jnp.array([[s * r]]))

        J = jax.scipy.linalg.block_diag(*blocks)

        X = jax.random.normal(k3, (dim_x, dim_x))
        Q, _ = jnp.linalg.qr(X)
        A = Q @ J @ Q.T

    else:
        raise ValueError(f"Unknown structure: {structure}")

    C = jax.random.normal(k4, (dim_y, dim_x)) / jnp.sqrt(dim_x)

    return A, C


@partial(jax.jit, static_argnums=(1, 2))
def generate_sequences(key, batch_size, T, A, C, noise_std):
    """Generate LDS sequences using jax.lax.scan."""
    dim_x = A.shape[0]
    dtype = A.dtype

    def step(x_t, k):
        eps = jax.random.normal(k, x_t.shape) * noise_std
        x_next = A @ x_t + eps
        y_next = C @ x_next
        return x_next, (x_next, y_next)

    keys = jax.random.split(key, batch_size)

    def simulate_one(k):
        ks = jax.random.split(k, T)
        x0 = jnp.zeros(dim_x, dtype=dtype)
        _, (xs, ys) = jax.lax.scan(step, x0, ks)
        return xs, ys

    xs, ys = jax.vmap(simulate_one)(keys)
    return xs, ys


class LDSDataset(BaseDataset):
    """LDS trajectory prediction with rich system generation and KF baseline."""

    def __init__(
        self,
        dim_x: int = 64,
        dim_y: int = 2,
        lambda_val: float = 0.9,
        structure: str = "dense",
        sigma: float = 1.0,
        sequence_length: int = 256,
        rng: jax.Array,
    ):
        super().__init__(name="lds")
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.sigma = sigma
        self.sequence_length = sequence_length

        # Generate system
        key, sys_key = jax.random.split(rng)
        A, C = generate_system_parameters(sys_key, dim_x, dim_y, lambda_val, structure)
        self.A = A
        self.C = C

        # Compute KF baseline MSE
        key, val_key = jax.random.split(key)
        val_T = 1024
        val_bs = 128
        _, val_ys = generate_sequences(val_key, val_bs, val_T, A, C, sigma)

        kf_filter_vmap = jax.vmap(kalman_filter, in_axes=(0, None, None, None))
        kf_preds, _ = kf_filter_vmap(val_ys, A, C, sigma)
        kf_err = val_ys - kf_preds
        self.kf_mse = float(jnp.mean(jnp.square(kf_err)[:, 64:]))
        self.loss_scale = 1.0 / self.kf_mse

        print(f"Baseline KF MSE: {self.kf_mse:.6f}")

    def get_batch(self, rng: jax.Array, batch_size: int) -> dict:
        """Return batch. inputs/targets are observation sequences."""
        _, ys = generate_sequences(rng, batch_size, self.sequence_length, self.A, self.C, self.sigma)
        # inputs and targets are both ys (next-step prediction via loss alignment)
        mask = jnp.ones(ys.shape[:2], dtype=jnp.float32)
        return {
            "inputs": ys,
            "targets": ys,
            "mask": mask,
            "loss_scale": self.loss_scale,
        }

    def get_eval_batch(self, rng: jax.Array, batch_size: int) -> dict:
        return self.get_batch(rng, batch_size)

    @staticmethod
    def loss_fn(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
        """MSE loss with patching alignment.

        logits: (B, N+1, P*dim_y) where N = T/P patches.
        targets: (B, T, dim_y)

        Reshape logits to (B, (N+1)*P, dim_y), take [:, 1:T] vs targets[:, 1:]
        for next-step prediction MSE.
        """
        B, T, dim_y = targets.shape
        # logits may have different seq dim due to patching + zero prepend
        _, n_plus_1, pd = logits.shape
        patch_size = pd // dim_y

        # Reshape logits from (B, N+1, P*dim_y) -> (B, (N+1)*P, dim_y)
        preds = logits.reshape(B, n_plus_1 * patch_size, dim_y)

        # Align: preds[:, 1:T] predicts targets[:, 1:]
        preds_aligned = preds[:, 1:T]
        targets_aligned = targets[:, 1:]

        return jnp.mean((preds_aligned - targets_aligned) ** 2)

    def compute_metrics(
        self, logits: jax.Array, targets: jax.Array, mask: jax.Array
    ) -> dict[str, float]:
        raw_mse = float(self.loss_fn(logits, targets, mask))
        return {
            f"{self.name}/mse": raw_mse,
            f"{self.name}/score": raw_mse / self.kf_mse,
        }

    def evaluate_rollouts(self, state, generate_fn, config) -> dict:
        """Evaluate model and KF rollouts at multiple horizons."""
        max_h = 64
        eval_horizons = list(range(1, max_h + 1))
        warmup_len = config.eval.warmup_len
        n_eval = config.eval.n_eval
        eval_batch_size = 256
        num_eval_batches = max(1, n_eval // eval_batch_size)
        seq_len = warmup_len + max_h

        get_final_state = jax.vmap(kalman_filter_final_state, in_axes=(0, None, None, None))

        model_errors = {h: [] for h in eval_horizons}
        kf_errors = {h: [] for h in eval_horizons}

        key = jax.random.PRNGKey(42)

        for i in range(num_eval_batches):
            key, subkey = jax.random.split(key)
            _, ys_eval = generate_sequences(subkey, eval_batch_size, seq_len, self.A, self.C, self.sigma)

            warmup = ys_eval[:, :warmup_len]
            truth = ys_eval[:, warmup_len:]

            # Model rollout via generate()
            generated = generate_fn(
                state, warmup,
                max_new_tokens=max_h,
                rng_key=subkey,
                mode="continuous",
            )  # (B, warmup_len + max_h, P*dim_y)

            # Extract predictions after warmup
            model_preds = generated[:, warmup_len:, :self.dim_y]

            # KF rollout
            x_last = get_final_state(warmup, self.A, self.C, self.sigma)
            kf_preds_list = []
            x_curr = x_last
            for _ in range(max_h):
                x_curr = x_curr @ self.A.T
                y_pred = x_curr @ self.C.T
                kf_preds_list.append(y_pred[:, None, :])
            kf_rollout = jnp.concatenate(kf_preds_list, axis=1)

            for h in eval_horizons:
                idx = h - 1
                m_err = jnp.mean(jnp.square(model_preds[:, idx] - truth[:, idx]), axis=-1)
                k_err = jnp.mean(jnp.square(kf_rollout[:, idx] - truth[:, idx]), axis=-1)
                model_errors[h].extend(m_err.tolist())
                kf_errors[h].extend(k_err.tolist())

        metrics = {}
        for h in eval_horizons:
            mean_model = np.mean(model_errors[h])
            mean_kf = np.mean(kf_errors[h])
            ratio = mean_model / mean_kf
            metrics[f"eval/score_t+{h}"] = ratio

        return metrics
