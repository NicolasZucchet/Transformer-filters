import argparse
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax.training import train_state
import wandb
import numpy as np
import time

from src.data import generate_system_parameters, generate_sequences
from src.model import TransformerFilter
from src.kalman import kalman_filter, kalman_filter_final_state
from src.eval import evaluate_model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--lambda_val", type=float, default=0.9)
    parser.add_argument("--structure", type=str, default="dense", choices=["dense", "diagonal", "scalar"])
    parser.add_argument("--patch_size", type=int, default=1)
    parser.add_argument("--dim_x", type=int, default=64)
    parser.add_argument("--dim_y", type=int, default=2)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--n_eval", type=int, default=8192)
    parser.add_argument("--wandb_project", type=str, default="Transformer-filters")
    parser.add_argument("--remove_inverse_norm", action="store_true", help="If set, skip the inverse normalization at the end of the network.")
    return parser.parse_args()

def main():
    args = parse_args()
    wandb.init(project=args.wandb_project, config=vars(args))
    
    key = jax.random.PRNGKey(args.seed)
    
    # 1. Generate System
    key, subkey = jax.random.split(key)
    A, C = generate_system_parameters(subkey, args.dim_x, args.dim_y, args.lambda_val, args.structure)
    
    # 2. Estimate Kalman Filter Baseline Variance
    key, subkey = jax.random.split(key)
    val_T = 1024
    val_bs = 128
    val_xs, val_ys = generate_sequences(subkey, val_bs, val_T, A, C, args.sigma)
    
    # Prepare KF args
    if jnp.iscomplexobj(A):
        KF_A = jnp.block([[A.real, -A.imag], [A.imag, A.real]])
        KF_C = jnp.block([[C.real, -C.imag]])
        KF_Q = args.sigma / jnp.sqrt(2)
    else:
        KF_A = A
        KF_C = C
        KF_Q = args.sigma
        
    kf_filter_vmap = jax.vmap(kalman_filter, in_axes=(0, None, None, None))
    kf_preds, kf_stats = kf_filter_vmap(val_ys, KF_A, KF_C, KF_Q)
    kf_err = val_ys - kf_preds
    kf_mse = jnp.mean(jnp.square(kf_err)[:, 64:])
    print(f"Baseline KF MSE: {kf_mse}")
    wandb.log({"baseline_kf_mse": float(kf_mse)})
    
    # 3. Model Setup
    model = TransformerFilter(patch_size=args.patch_size, dim_y=args.dim_y, remove_inverse_norm=args.remove_inverse_norm)
    
    dummy_input = jnp.zeros((1, args.seq_len, args.dim_y))
    key, subkey = jax.random.split(key)
    variables = model.init(subkey, dummy_input)
    
    tx = optax.adamw(learning_rate=args.lr)
    state = train_state.TrainState.create(apply_fn=model.apply, params=variables['params'], tx=tx)
    
    # 4. Training Loop
    @jax.jit
    def train_step(state, batch_y):
        def loss_fn(params):
            preds = model.apply({'params': params}, batch_y)
            preds_aligned = preds[:, :-1]
            targets_aligned = batch_y[:, 1:]
            
            loss = jnp.mean(jnp.square(preds_aligned - targets_aligned))
            return loss / kf_mse
            
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss

    print("Starting training...")
    start_time = time.time()
    
    eval_interval = max(1, args.steps // 4)
    
    for step in range(args.steps):
        key, subkey = jax.random.split(key)
        xs, ys = generate_sequences(subkey, args.batch_size, args.seq_len, A, C, args.sigma)
        
        state, loss = train_step(state, ys)
        
        if step % 50 == 0:
            wandb.log({"train_loss": float(loss), "step": step})
            
        if (step + 1) % eval_interval == 0:
            print(f"Evaluating at step {step + 1}...")
            evaluate_model(
                model, state.params, A, C, args.sigma, KF_A, KF_C, KF_Q, 
                args.seed, args.n_eval, 256, seq_len=128, warmup_len=64, 
                patch_size=args.patch_size, step=step + 1
            )
            
    print(f"Training finished in {time.time() - start_time:.2f}s")
    
    wandb.finish()

if __name__ == "__main__":
    main()
