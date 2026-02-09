"""Entry point for transformer experiments."""
import jax
import wandb
from src.config import Config, parse_config
from src.data.factory import create_dataset
from src.model.factory import create_model
from src.train import create_optimizer, create_train_state, train_loop


def main():
    config = parse_config(Config, description="Transformer Filters")
    print(config)

    # Initialize WandB
    wandb.init(
        project=config.wandb_project,
        mode=config.wandb_mode,
        config=vars_nested(config),
    )

    rng = jax.random.PRNGKey(config.seed)
    rng, data_rng, model_rng = jax.random.split(rng, 3)

    # Create dataset
    dataset = create_dataset(config.data, data_rng)

    # Log KF baseline if available
    if hasattr(dataset, 'kf_mse'):
        wandb.log({"baseline_kf_mse": dataset.kf_mse}, step=0)

    # Create model
    model = create_model(config.model, config.data)

    # Create optimizer and train state
    optimizer = create_optimizer(config.train)
    state = create_train_state(model, config, model_rng, dataset, optimizer)

    print(f"Model parameters: {sum(x.size for x in jax.tree.leaves(state.params)):,}")

    # Train
    state = train_loop(state, dataset, config)

    wandb.finish()


def vars_nested(obj):
    """Convert nested dataclasses to dict."""
    import dataclasses
    if dataclasses.is_dataclass(obj):
        return {k: vars_nested(v) for k, v in dataclasses.asdict(obj).items()}
    return obj


if __name__ == "__main__":
    main()
