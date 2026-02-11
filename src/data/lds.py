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
def generate_sequences(key, batch_size, T, A, C, noise_std,
                       obs_noise_std=0.0, b=None, x0_std=0.0):
    """Generate LDS sequences using jax.lax.scan.

    Args:
        noise_std: process noise std (sigma)
        obs_noise_std: observation noise std (added to y)
        b: optional affine bias (dx,)
        x0_std: initial state std (x0 ~ N(0, x0_std^2 I))
    """
    dim_x = A.shape[0]
    dtype = A.dtype
    bias = b if b is not None else jnp.zeros(dim_x, dtype=dtype)

    def step(x_t, k):
        k1, k2 = jax.random.split(k)
        eps = jax.random.normal(k1, x_t.shape) * noise_std
        x_next = A @ x_t + bias + eps
        obs_noise = jax.random.normal(k2, (C.shape[0],)) * obs_noise_std
        y_next = C @ x_next + obs_noise
        return x_next, (x_next, y_next)

    keys = jax.random.split(key, batch_size)

    def simulate_one(k):
        k1, k2 = jax.random.split(k)
        ks = jax.random.split(k1, T)
        x0 = jax.random.normal(k2, (dim_x,), dtype=dtype) * x0_std
        _, (xs, ys) = jax.lax.scan(step, x0, ks)
        return xs, ys

    xs, ys = jax.vmap(simulate_one)(keys)
    return xs, ys


class LDSDataset(BaseDataset):
    """LDS trajectory prediction with rich system generation and KF baseline."""

    def __init__(
        self,
        *,
        dim_x: int = 64,
        dim_y: int = 2,
        lambda_val: float = 0.9,
        structure: str = "dense",
        sigma: float = 1.0,
        sequence_length: int = 256,
        obs_noise_std: float = 0.0,
        x0_std: float = 0.0,
        eval_sequence_length: int = 0,
        skip_kf: bool = False,
        rng: jax.Array,
    ):
        super().__init__(name="lds")
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.sigma = sigma
        self.sequence_length = sequence_length
        self.obs_noise_std = obs_noise_std
        self.x0_std = x0_std
        self.eval_sequence_length = eval_sequence_length if eval_sequence_length > 0 else sequence_length
        self.skip_kf = skip_kf
        self.b = None  # affine bias, set by subclasses

        # Generate system
        key, sys_key = jax.random.split(rng)
        self._setup_system(sys_key, dim_x, dim_y, lambda_val, structure)

        # Compute KF baseline MSE
        if skip_kf:
            self.kf_mse = 1.0
            self.loss_scale = 1.0
        else:
            self._compute_kf_baseline(key)

    def _setup_system(self, key, dim_x, dim_y, lambda_val, structure):
        """Set self.A, self.C (and optionally self.b). Override in subclasses."""
        A, C = generate_system_parameters(key, dim_x, dim_y, lambda_val, structure)
        self.A = A
        self.C = C

    def _compute_kf_baseline(self, key):
        """Compute KF baseline MSE and loss_scale."""
        # Build P0 for KF
        P0 = None
        if self.x0_std > 0:
            P0 = (self.x0_std ** 2) * jnp.eye(self.dim_x)

        # Cache vmap'd KF functions
        self._kf_filter_vmap = jax.jit(jax.vmap(
            kalman_filter, in_axes=(0, None, None, None, None, None, None)))
        self._kf_final_state_vmap = jax.jit(jax.vmap(
            kalman_filter_final_state, in_axes=(0, None, None, None, None, None, None)))

        val_T = 1024
        val_bs = 128
        _, val_ys = generate_sequences(
            key, val_bs, val_T, self.A, self.C, self.sigma,
            obs_noise_std=self.obs_noise_std, b=self.b, x0_std=self.x0_std)

        R_std = self.obs_noise_std if self.obs_noise_std > 0 else 1e-4
        kf_preds, _ = self._kf_filter_vmap(
            val_ys, self.A, self.C, self.sigma, R_std, self.b, P0)
        kf_err = val_ys - kf_preds
        self.kf_mse = float(jnp.mean(jnp.square(kf_err)[:, 64:]))
        self.loss_scale = 1.0 / self.kf_mse

        print(f"Baseline KF MSE: {self.kf_mse:.6f}")

    def _generate(self, rng, batch_size, T):
        """Generate sequences with current system parameters."""
        return generate_sequences(
            rng, batch_size, T, self.A, self.C, self.sigma,
            obs_noise_std=self.obs_noise_std, b=self.b, x0_std=self.x0_std)

    def get_batch(self, rng: jax.Array, batch_size: int) -> dict:
        """Return batch. inputs/targets are observation sequences."""
        _, ys = self._generate(rng, batch_size, self.sequence_length)
        mask = jnp.ones(ys.shape[:2], dtype=jnp.float32)
        return {
            "inputs": ys,
            "targets": ys,
            "mask": mask,
            "loss_scale": self.loss_scale,
        }

    def get_eval_batch(self, rng: jax.Array, batch_size: int) -> dict:
        """Return eval batch (may use different sequence length)."""
        _, ys = self._generate(rng, batch_size, self.eval_sequence_length)
        mask = jnp.ones(ys.shape[:2], dtype=jnp.float32)
        return {
            "inputs": ys,
            "targets": ys,
            "mask": mask,
            "loss_scale": self.loss_scale,
        }

    @staticmethod
    def loss_fn(logits: jax.Array, targets: jax.Array, mask: jax.Array) -> jax.Array:
        """MSE loss with patching alignment.

        logits: (B, N+1, P*dim_y) where N = T/P patches.
        targets: (B, T, dim_y)

        Reshape logits to (B, (N+1)*P, dim_y), take [:, 1:T] vs targets[:, 1:]
        for next-step prediction MSE.
        """
        B, T, dim_y = targets.shape
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
        import wandb

        max_h = config.eval.rollout_steps
        warmup_len = config.eval.warmup_len
        n_eval = config.eval.n_eval
        eval_batch_size = 256
        num_eval_batches = max(1, n_eval // eval_batch_size)
        seq_len = warmup_len + max_h

        A_T = self.A.T
        C_T = self.C.T
        bias = self.b if self.b is not None else jnp.zeros(self.dim_x)

        # Build P0 for KF (only if needed)
        if not self.skip_kf:
            P0 = None
            if self.x0_std > 0:
                P0 = (self.x0_std ** 2) * jnp.eye(self.dim_x)
            R_std = self.obs_noise_std if self.obs_noise_std > 0 else 1e-4

        all_model_mse = []
        all_kf_mse = []
        # Store first batch for trajectory visualization
        viz_truth = None
        viz_model = None
        viz_kf = None

        key = jax.random.PRNGKey(42)

        for i in range(num_eval_batches):
            key, subkey = jax.random.split(key)
            _, ys_eval = self._generate(subkey, eval_batch_size, seq_len)

            warmup = ys_eval[:, :warmup_len]
            truth = ys_eval[:, warmup_len:]

            # Model rollout via generate()
            generated = generate_fn(
                state, warmup,
                max_new_tokens=max_h,
                rng_key=subkey,
                mode="continuous",
            )
            model_preds = generated[:, warmup_len:, :self.dim_y]

            model_mse = jnp.mean(jnp.square(model_preds - truth), axis=-1)
            all_model_mse.append(model_mse)

            if not self.skip_kf:
                # KF rollout
                x_last = self._kf_final_state_vmap(
                    warmup, self.A, self.C, self.sigma, R_std, self.b, P0)

                def kf_step(x, _):
                    x_next = x @ A_T + bias
                    y_pred = x_next @ C_T
                    return x_next, y_pred

                _, kf_rollout = jax.lax.scan(kf_step, x_last, None, length=max_h)
                kf_rollout = jnp.moveaxis(kf_rollout, 0, 1)

                kf_mse = jnp.mean(jnp.square(kf_rollout - truth), axis=-1)
                all_kf_mse.append(kf_mse)

            if i == 0:
                viz_truth = np.array(truth[:4])
                viz_model = np.array(model_preds[:4])
                if not self.skip_kf:
                    viz_kf = np.array(kf_rollout[:4])

        all_model_mse = jnp.concatenate(all_model_mse, axis=0)
        mean_model = np.array(jnp.mean(all_model_mse, axis=0))

        metrics = {}
        if self.skip_kf:
            for h in range(1, max_h + 1):
                metrics[f"eval/score_t+{h}"] = float(mean_model[h - 1])
        else:
            all_kf_mse = jnp.concatenate(all_kf_mse, axis=0)
            mean_kf = np.array(jnp.mean(all_kf_mse, axis=0))
            for h in range(1, max_h + 1):
                metrics[f"eval/score_t+{h}"] = float(mean_model[h - 1] / mean_kf[h - 1])

        # Log trajectory visualizations
        self._log_rollout_trajectories(viz_truth, viz_model, viz_kf, max_h)

        return metrics

    @staticmethod
    def _log_rollout_trajectories(truth, model_preds, kf_preds, max_h):
        """Log example rollout trajectories to wandb as line plots."""
        import wandb

        n_examples = truth.shape[0]
        dim_y = truth.shape[2]
        timesteps = list(range(1, max_h + 1))

        plots = {}
        for dim in range(min(dim_y, 2)):  # log first 2 output dims
            for i in range(n_examples):
                ys = [
                    [float(truth[i, t-1, dim]) for t in timesteps],
                    [float(model_preds[i, t-1, dim]) for t in timesteps],
                ]
                keys = ["truth", "model"]
                if kf_preds is not None:
                    ys.append([float(kf_preds[i, t-1, dim]) for t in timesteps])
                    keys.append("kf")
                plots[f"eval/rollout_y{dim}_ex{i}"] = wandb.plot.line_series(
                    xs=timesteps,
                    ys=ys,
                    keys=keys,
                    title=f"Rollout dim={dim} example={i}",
                    xname="horizon",
                )
        wandb.log(plots, commit=False)
