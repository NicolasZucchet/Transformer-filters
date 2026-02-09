"""Training utilities and training loop."""
from typing import Any, Callable
from functools import partial
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
from src.config import Config, TrainConfig


class TrainState(train_state.TrainState):
    """Training state with additional RNG field."""
    rng: jax.Array


def create_optimizer(train_config: TrainConfig) -> optax.GradientTransformation:
    """Create optimizer with warmup + cosine decay and gradient clipping."""
    # Warmup + cosine decay schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=train_config.learning_rate,
        transition_steps=train_config.warmup_steps,
    )
    cosine_fn = optax.cosine_decay_schedule(
        init_value=train_config.learning_rate,
        decay_steps=max(1, train_config.total_steps - train_config.warmup_steps),
    )
    schedule = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[train_config.warmup_steps],
    )

    return optax.chain(
        optax.clip_by_global_norm(train_config.grad_clip),
        optax.adamw(learning_rate=schedule, weight_decay=train_config.weight_decay),
    )


def create_train_state(
    model: nn.Module,
    config: Config,
    rng: jax.Array,
    dataset: Any,
    optimizer: optax.GradientTransformation,
) -> TrainState:
    """Initialize parameters and create train state."""
    rng, init_rng = jax.random.split(rng)

    # Create dummy input based on task type
    if config.data.task_type == "bigram":
        dummy_input = jnp.ones((1, config.data.sequence_length), dtype=jnp.int32)
    elif config.data.task_type == "lds":
        dummy_input = jnp.ones((1, config.data.sequence_length, config.data.dim_y), dtype=jnp.float32)
    else:
        raise ValueError(f"Unknown task_type: {config.data.task_type}")

    init_rngs = {"params": init_rng, "dropout": init_rng}
    variables = model.init(init_rngs, dummy_input, deterministic=True)
    params = variables["params"]

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        rng=rng,
    )


@partial(jax.jit, static_argnums=(2,))
def train_step(
    state: TrainState,
    batch: dict,
    loss_fn: Callable,
) -> tuple[TrainState, dict]:
    """Single training step. loss_fn is a static argument for JIT."""
    dropout_rng = jax.random.fold_in(state.rng, state.step)

    def compute_loss(params):
        logits = state.apply_fn(
            {"params": params}, batch["inputs"],
            deterministic=False, rngs={"dropout": dropout_rng},
        )
        loss = loss_fn(logits, batch["targets"], batch["mask"])
        # Apply loss_scale for gradient scaling (e.g. 1/kf_mse for LDS)
        loss_scale = batch.get("loss_scale", 1.0)
        return loss * loss_scale

    loss, grads = jax.value_and_grad(compute_loss)(state.params)
    new_state = state.apply_gradients(grads=grads)

    return new_state, {"loss": loss}


@partial(jax.jit, static_argnums=(0,))
def eval_forward(apply_fn, params, inputs):
    """JIT-compiled forward pass for evaluation."""
    return apply_fn({"params": params}, inputs, deterministic=True)


def eval_model(state: TrainState, dataset, config: Config) -> dict:
    """Evaluate model on dataset (teacher-forced)."""
    rng = jax.random.PRNGKey(0)
    batch = dataset.get_eval_batch(rng, config.data.eval_batch_size)
    logits = eval_forward(state.apply_fn, state.params, batch["inputs"])
    metrics = dataset.compute_metrics(logits, batch["targets"], batch["mask"])
    return metrics


def train_loop(state: TrainState, dataset, config: Config) -> TrainState:
    """Main training loop with logging and evaluation."""
    import wandb

    loss_fn = dataset.loss_fn
    nan_check_interval = max(1, config.train.total_steps // 100)

    for step in range(config.train.total_steps):
        rng, batch_rng = jax.random.split(state.rng)
        state = state.replace(rng=rng)
        batch = dataset.get_batch(batch_rng, config.data.batch_size)

        state, train_metrics = train_step(state, batch, loss_fn)

        # NaN check
        if (step + 1) % nan_check_interval == 0 and jnp.isnan(train_metrics["loss"]):
            print(f"NaN loss at step {step + 1}, stopping.")
            break

        # Logging
        if (step + 1) % config.train.log_interval == 0:
            wandb.log({"train/loss": float(train_metrics["loss"])}, step=step + 1)

        # Evaluation
        if step == 0 or (step + 1) % config.eval.eval_interval == 0:
            eval_metrics = eval_model(state, dataset, config)

            # Rollout evaluation
            if config.eval.do_rollout:
                from src.model.inference import generate
                rollout_metrics = dataset.evaluate_rollouts(
                    state, generate, config,
                )
                eval_metrics.update(rollout_metrics)

            wandb.log(eval_metrics, step=step + 1)

            print(f"Step {step + 1}/{config.train.total_steps}")
            for k, v in eval_metrics.items():
                if isinstance(v, (int, float)):
                    print(f"  {k}: {v:.4f}")

    return state
