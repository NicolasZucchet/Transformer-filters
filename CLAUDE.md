# Transformer Filters

## Project Overview

Experimental codebase for Transformer-based filtering on synthetic linear dynamical systems (LDS). Also supports bigram language model task. Built with JAX/Flax/Optax, WandB for logging, `uv` for dependency management.

## Project Structure

```
.
├── pyproject.toml
├── src/
│   ├── main.py              # Entry point
│   ├── config.py            # Top-level Config + auto CLI generation from dataclasses
│   ├── train.py             # TrainState, optimizer, train_step, train_loop
│   ├── kalman.py            # Kalman filter (baseline)
│   ├── model/
│   │   ├── config.py        # ModelConfig (+ patch_size)
│   │   ├── factory.py       # create_model() -> SequenceModel
│   │   ├── backbone.py      # TransformerBackbone (RoPE/learned PE, KV cache)
│   │   ├── layers.py        # EmbeddingInput, LinearInput, PatchingInput
│   │   ├── heads.py         # ClassificationHead, RegressionHead
│   │   └── inference.py     # generate() with KV cache (discrete + continuous)
│   └── data/
│       ├── config.py        # DataConfig (LDS params: dim_x, dim_y, lambda_val, structure, sigma)
│       ├── factory.py       # create_dataset()
│       ├── base.py          # BaseDataset ABC
│       ├── metrics.py       # accuracy, CE, perplexity, MSE, per-step MSE
│       ├── bigram.py        # BigramDataset (random Markov chain)
│       └── lds.py           # LDSDataset (rich LDS with KF baseline normalization)
├── scripts/
│   └── start_euler_sweep.sh
├── sweeps/
│   ├── example_bigram.yaml
│   ├── exp1_eigenvalues.yaml
│   ├── exp2_matrix_structure.yaml
│   ├── exp3_patching.yaml
│   └── exp4_obs_dim.yaml
└── analysis/
    ├── common.py
    ├── analyze_exp1.py
    ├── analyze_exp2.py
    ├── analyze_exp3.py
    ├── analyze_exp4.py
    └── analyze_debug_inverse_norm.py
```

## Running

```bash
# Bigram task
uv run python -m src.main --data.task_type bigram

# LDS task
uv run python -m src.main --data.task_type lds

# LDS with patching
uv run python -m src.main --data.task_type lds --model.patch_size 4

# With rollout evaluation
uv run python -m src.main --data.task_type lds --eval.do_rollout true

# Override any config field with dot notation
uv run python -m src.main --model.model_dim 128 --model.num_layers 2 --train.learning_rate 0.0003
```

## Architecture

### Config System
- Dataclass-based configs with auto-generated CLI via `config.py`
- All fields support `--dot.notation` overrides, no manual argparse registration needed
- Hierarchical: `Config` composes `ModelConfig`, `DataConfig`, `TrainConfig`, `EvalConfig`

### Model: SequenceModel = InputLayer + TransformerBackbone + OutputHead
- **InputLayer**: `EmbeddingInput` (bigram), `PatchingInput` (LDS — patches + causal zero-prepend + linear projection)
- **TransformerBackbone**: Pre-LayerNorm blocks, RoPE or learned positional encoding, KV cache for generation
- **OutputHead**: `ClassificationHead` (bigram) or `RegressionHead` (LDS, output_dim = patch_size * dim_y)
- `factory.py` selects components based on `data_config.task_type`

### Datasets
Each dataset provides:
- `get_batch()` / `get_eval_batch()` returning `{inputs, targets, mask}` (+ `loss_scale` for LDS)
- `loss_fn(logits, targets, mask)` — must be a static function (used as JIT static arg, separate compilation per loss)
- `compute_metrics()` for teacher-forced evaluation, prefixed with `{dataset_name}/`
- `evaluate_rollouts()` for autoregressive generation evaluation (optional override)

### LDS-specific
- **System generation**: Dense (rotation blocks + similarity transform), diagonal, or scalar A matrices. C is random projection.
- **Loss normalization**: `loss_scale = 1/kf_mse` included in batch and applied inside gradient computation
- **Patching**: `PatchingInput` reshapes `(B, T, dim_y)` → `(B, T/P, P*dim_y)`, prepends zero patch, projects to model_dim
- **Loss alignment**: Model output `(B, N+1, P*dim_y)` reshaped to `(B, (N+1)*P, dim_y)`, `preds[:, 1:T]` vs `targets[:, 1:]`
- **Rollout eval**: Uses `generate()` for model + KF rollout, logs `eval/score_t+{h}` for horizons 1..64

### Training
- Warmup + cosine decay schedule
- Gradient clipping (default 1.0)
- AdamW optimizer
- Flax Dropout requires `rngs={"dropout": key}` when `deterministic=False`

### WandB Integration
- Initialize in `main.py` with project and run names
- Log training metrics every `train.log_interval` steps
- Log evaluation metrics every `eval.eval_interval` steps
- Hierarchical logging:
  - `train/{metric}` — training loss
  - `{dataset_name}/{metric}` — teacher-forced evaluation
  - `eval/score_t+{h}` — rollout evaluation (LDS)

## Development Guidelines

### Code Style
- **Compact and factorized**: Prefer compact, factorized code — extract shared logic into helpers, avoid duplication
- **Type hints**: Use JAX types (`jax.Array`, etc.) and standard Python types
- **Docstrings**: Google style, minimal but clear
- **Imports**: Group stdlib, third-party, and local imports
- **JAX best practices**:
  - Use `jax.jit` for performance
  - Keep functions pure when possible
  - Use `jax.tree` for pytree operations

### Adding New Datasets

1. Create `src/data/new_task.py`, inherit from `BaseDataset`
2. Implement: `get_batch`, `get_eval_batch`, `loss_fn` (static), `compute_metrics`
3. Optionally implement `evaluate_rollouts`
4. Register in `src/data/factory.py`
5. Add task-specific config fields to `src/data/config.py`
6. Update `src/train.py:create_train_state` dummy input shape if needed
7. Update `src/model/factory.py` input/head selection

### Checking Sweep Status

To determine if a sweep is fully complete for the project:

1. **Check Sweep State**: Done if state is `FINISHED`, `CANCELED`, or `FAILED`.
2. **Check Active Runs**: If state is `RUNNING`, it is only done if no runs are `running` or `pending`.

## Sweep Configuration

### `start_euler_sweep.sh`
`SERVER_HOST`, `WORKSPACE_DIR`, and `WANDB_PROJECT` are configured at the top of the script. Update them before first use.

```bash
bash scripts/start_euler_sweep.sh sweeps/exp1_eigenvalues.yaml
```

## Notes

- No checkpointing or model saving currently implemented
- No testing infrastructure
- Focus on rapid experimentation with synthetic data
- Must run with `python -m src.main`, not `python src/main.py` (the latter breaks `from src.X` imports)

## Git Commit Guidelines

- Use shortest possible commit messages
- No additional info beyond the commit message itself (in particular no mention that the commit was generated by Claude)
