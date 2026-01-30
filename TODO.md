# Transformer Filters Project Status

## Done ✅

### 1. Infrastructure & Setup
- [x] Initialized project with `uv` and managed dependencies (`jax`, `flax`, `optax`, `wandb`, etc.).
- [x] Set up JAX-compatible project structure with `src/` package.
- [x] Configured Git and pushed initial implementation to remote.

### 2. Data Generation (`src/data.py`)
- [x] Implemented State-Space Model ($x_{t+1} = Ax_t + \varepsilon$, $y_t = Cx_t$).
- [x] Implemented eigenvalue sampling in range $[\lambda/2, \lambda]$.
- [x] Support for **Dense (Real)** and **Diagonal (Complex)** $A$ matrices.
- [x] Proper normalization for $C$ (fan_out).

### 3. Baseline implementation (`src/kalman.py`)
- [x] Implemented Kalman Filter in JAX.
- [x] Support for calculating steady-state MSE for loss normalization.
- [x] Added helper for extracting final posterior state for rollout initialization.

### 4. Model Architecture (`src/model.py`)
- [x] Standard Transformer (4 layers, $d_{model}=256$, 4 heads).
- [x] **RoPE** (Rotary Positional Embeddings) implementation.
- [x] **Patching** logic (configurable patch size).
- [x] **Causal RevIN** (Reversible Instance Normalization) to handle non-stationarity while respecting causality.

### 5. Training & Evaluation (`src/train.py`)
- [x] Training loop using next-token prediction.
- [x] Loss normalization by Kalman Filter variance.
- [x] **Rollout Evaluation**:
    - [x] 64-step warmup context.
    - [x] Multi-step prediction ($t+2, t+4, \dots, t+64$).
    - [x] Parallel rollout for 8192 sequences.
    - [x] Metric normalization by KF performance.
- [x] WandB integration for all metrics and configurations.

### 6. Experiment Design
- [x] `sweep_exp1_eigenvalues.yaml`: **Exp 1** (Eigenvalue sweep).
- [x] `sweep_exp2_matrix_structure.yaml`: **Exp 2** (Diagonal vs Dense).
- [x] `sweep_exp3_patching.yaml`: **Exp 3** (Patching interactions).
- [x] `sweep_exp4_obs_dim.yaml`: **Exp 4** (Observation dimension impact).

---

## Left To Do ⏳

### 1. Execution
- [ ] Run Exp 1: `./start_sweep.sh sweep_exp1_eigenvalues.yaml`
- [ ] Run Exp 2: `./start_sweep.sh sweep_exp2_matrix_structure.yaml`
- [ ] Run Exp 3: `./start_sweep.sh sweep_exp3_patching.yaml`
- [ ] Run Exp 4: `./start_sweep.sh sweep_exp4_obs_dim.yaml`

### 2. Verification & Analysis
- [ ] Monitor WandB to ensure all runs (10-100 per experiment) complete successfully.
- [ ] **Hypothesis Testing**:
    - [ ] Analyze if **Diagonal A** is indeed easier to learn.
    - [ ] Verify if **Patching** importance scales with $\lambda$.
    - [ ] Confirm if **Lower dim_y** increases the necessity of patching.
- [ ] Identify and document "Easy", "Medium", and "Hard" $\lambda$ settings based on results.

### 3. Refinement
- [ ] Adjust Learning Rates if the current grid ([1e-3, 1e-4]) doesn't cover the optimal range for all configurations.
- [ ] Increase seed count if results are too noisy.
