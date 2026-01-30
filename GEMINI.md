# Transformer Filters: Machine Learning Experiment Specification

## Objective

Design and implement machine learning experiments to understand how Transformers solve time-series next-token prediction tasks.

## Data Generation Process

Generate a dataset using a state-space model:

- Initialize: `x_0 = 0`
- State evolution: `x_{t+1} = A * x_t + ε` where `ε ~ N(0, σ² * I)`
- Observation: `y_t = C * x_t` (use fan_out normalization for C)
- Task: Given `y_{1:t}`, predict `y_t`

## Configurable Parameters

- **Sequence length T**: 256 (default)
- **Latent dimension** (dim of x): 64 (default)
- **Observable dimension** (dim of y): 2 (default)
- **Noise std deviation σ**: 1.0 (default)
- **Eigenvalue λ**: 0.9 (default) - ensure A's eigenvalues are between λ/2 and λ

## Baseline Implementation

- Implement the Kalman filter as the optimal solution baseline
- Normalize L2 loss by the Kalman filter's loss variance for consistency across timescales

## Training and Evaluation

**Training**: Use next-token prediction loss (normalized by Kalman filter performance)

**Rollout Evaluation**:

- Warmup context window: 64 timesteps
- Measure performance at horizons: t+2, t+4, t+8, t+16, t+32, t+64
- Generate 8192 parallel rollouts (reduce if memory constrained - estimate memory usage for cluster)
- Normalize all metrics by Kalman filter performance
- Run rollout evaluation only after training completes

## Model Architecture

- **Input**: Normalization layer
- **Output**: Reversible instance normalization
- **Patching**: Combine consecutive timesteps (default=1, sweep up to 32 in powers of 2)
- **Positional encoding**: RoPE
- **Transformer**: 4 layers, d_model=256, 4 attention heads

## Experiments

### 1. Eigenvalue Sweep

Test λ values: [0.36, 0.68, 0.84, 0.92, 0.96, 0.98, 0.99]

- Hypothesis: Higher λ → harder learning
- Goal: Identify easy, medium, and hard settings for future use

### 2. Diagonal vs. Dense Matrix A

- Hypothesis: Diagonal A (with complex eigenvalues, ensuring real-valued y) is easier for Transformers
- Design a controlled experiment comparing diagonal vs. dense A matrices

### 3. Patching Importance vs. λ and Matrix Structure

- Hypothesis: Patching becomes more important as λ increases, but less important when A is diagonal
- Design experiment varying patch size across different λ values and matrix structures

### 4. Observation Dimension Impact

- Hypothesis: Smaller observation dimension increases task difficulty and patching importance
- Design experiment varying observable dimension and patch size

## Experimental Constraints

- **Runs per experiment**: 10-100
- **Training time**: 5-10 minutes per run on cluster
- **Learning rate**: Tune separately for each configuration
- **Seeds**: Use 2+ random seeds per configuration
- **Verification**: Confirm all runs complete before analysis

## Implementation Requirements

- **Framework**: JAX
- **Logging**: Weights & Biases (wandb)
- **Dependencies**: Manage with `uv`
- **Code quality**: Research-grade (functional, not production-polished)

## Execution Instructions

- **Debug runs**: Use `uv run`
- **Large-scale sweeps**: Use `start_sweep.sh` with wandb sweep YAML file
    - Note the sweep ID for later reference/continuation
    - Push code changes before running sweeps
    - Run test sweeps first to estimate cluster execution time
